import ctypes
import logging
import numpy as np
import random
from scipy.stats import binom
from functools import cache
import matplotlib.pyplot as plt
import matplotlib
import json
import math
import pandas as pd


def load_config(scheme):
    with open('./config.json', 'r') as file:
        config = json.load(file)
    return config.get(scheme, {})

def setup_logging(script,scheme,mvORfd):
    # Constants
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    # Remove all handlers associated with the root logger (clean up)
    DATE_FORMAT = '%m/%d/%Y %H:%M:%S'
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Set up new logging configuration
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=[
        logging.FileHandler(f'./logs/{script}_{scheme}_{mvORfd}.log'),
        #logging.StreamHandler()
    ])
    # Set Matplotlib's logger to only log warnings or higher
    matplotlib_logger = logging.getLogger(matplotlib.__name__)
    matplotlib_logger.setLevel(logging.WARNING)

def load_lib(lib_path):
    lib = ctypes.CDLL(lib_path)
    return lib

def load_rm_decoder(lib):
    rm_decode = lib.reed_muller_decode_one_block
    rm_decode.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.POINTER(ctypes.c_uint64)]
    rm_decode.restype = None
    return rm_decode


def load8_arr(in8): # uint8 to uint64
    inlen = len(in8)
    outlen = (inlen + 7) // 8  # Calculate needed length for uint64 array
    out64 = (ctypes.c_uint64 * outlen)()  # Create the output array
    
    index_in = 0
    index_out = 0
    
    # First copy by 8 bytes where possible, treating as big-endian
    while index_out < outlen and index_in + 8 <= inlen:
        # Convert bytes to a uint64 using big-endian, mimicking the load8 function
        out64[index_out] = int.from_bytes(in8[index_in:index_in + 8], 'big')
        index_in += 8
        index_out += 1

    # Handle the last few bytes if necessary
    if index_in < inlen:
        last_val = 0
        for i in range(inlen - index_in):
            # Shift and append each byte
            last_val <<= 8
            last_val |= in8[index_in + i]
        # Store the constructed last value in big-endian fashion
        out64[index_out] = last_val

    return out64

def bit_array_to_uint64(lib,vector,scheme):
   # Convert binary array to bytes
    byte_arr = bytearray()
    # int(len(vector)/8) gives the length of the correspondign byte array
    len_byte = math.ceil(len(vector)/8)
    for i in range(0, len_byte * 8, 8):
        # Convert each block of 8 binary digits to a byte
        byte = 0
        for bit in vector[i:i+8]:
            byte = (byte << 1) | bit
        byte_arr.append(byte)
    
    n_bytes = (int) (len_byte)

    # Create a ctypes array of type unsigned char and length n2/8
    byte_array_uint8 = (ctypes.c_ubyte * n_bytes)(*byte_arr)
    
    function_name = f"PQCLEAN_{scheme.upper()}_CLEAN_load8_arr"
    PQCLEAN_CLEAN_load8_arr = getattr(lib, function_name)

    PQCLEAN_CLEAN_load8_arr.argtypes = [
    ctypes.POINTER(ctypes.c_uint64),  # uint64_t *out64
    ctypes.c_size_t,                  # size_t outlen
    ctypes.POINTER(ctypes.c_uint8),   # const uint8_t *in8
    ctypes.c_size_t                   # size_t inlen
    ]
    PQCLEAN_CLEAN_load8_arr.restype = None
    
    len_64 = math.ceil(len(vector)/64)
    uint64_array = (ctypes.c_uint64 * len_64)()           # Output buffer
    PQCLEAN_CLEAN_load8_arr(uint64_array, len_64, byte_array_uint8, len_byte)

    return uint64_array


def bit_array_to_uint8(vector,n2): # get a ctypes array of n2/8 bytes representing the binary data.
    # Ensure the vector is exactly n2 bits
    if len(vector) != n2:
        raise ValueError("The binary vector must be of length n2/8.")
    
    # Convert binary array to bytes
    byte_arr = bytearray()
    for i in range(0, len(vector), 8):
        # Convert each block of 8 binary digits to a byte
        byte = 0
        for bit in vector[i:i+8]:
            byte = (byte << 1) | bit
        byte_arr.append(byte)
    
    n_bytes = (int) (n2 / 8)
    # Create a ctypes array of type unsigned char and length n2/8
    ctypes_arr = (ctypes.c_ubyte * n_bytes)(*byte_arr)
    
    return ctypes_arr


def uint64_to_bit(array):     # convert ctypes uint64 array to binary bit array
    np_uint64_array = np.frombuffer(array, dtype=np.uint64)
    # View the uint64 array as uint8
    np_uint8_array = np_uint64_array.view(np.uint8)
    # Use unpackbits to get the binary representation of each uint8
    binary_bits_array = np.unpackbits(np_uint8_array)
    return binary_bits_array


def uint8_to_bit(array):  # convert ctypes uint8 array to binary bit array
    # Create a NumPy array from the ctypes array buffer, dtype is already uint8
    np_uint8_array = np.frombuffer(array, dtype=np.uint8)

    # Use unpackbits to get the binary representation of each uint8
    binary_bits_array = np.unpackbits(np_uint8_array)
    return binary_bits_array

def sample_binary_vector(n2, min_weight, max_weight):
    # Sample a binary random vector of length n2 with Hamming weight between min_weight and max_weight.

    # Calculate actual Hamming weight bounds
    lower_bound = int(min_weight * n2)
    upper_bound = int(max_weight * n2)
    
    # Generate a random Hamming weight within the bounds
    weight = random.randint(lower_bound, upper_bound)
    
    # Create the vector
    vector = np.zeros(n2, dtype=int)
    vector[:weight] = 1
    np.random.shuffle(vector)
    
    return vector

def mutate_bit_by_bit(vector): # only flip one bit
    flip_index = random.randint(0, len(vector) - 1)
    # Flip the bit: XOR with 1 will toggle the bit at flip_index
    vector[flip_index] ^= 1
    return vector

def mutate2bits(vector): # flip two bits

    flip_indices = random.sample(range(len(vector)), 2)

    # Flip the bits at the selected indices: XOR with 1 will toggle the bits
    for flip_index in flip_indices:
        vector[flip_index] ^= 1
    return vector


# Global scheme data, distribution of u
scheme_data = {
    'hqc128': {'n2': 384, 'probabilities': [23.44, 34.38, 24.83, 11.77, 4.12, 1.45]},
    'hqc192': {'n2': 640, 'probabilities': [16.50, 30.00, 27.00, 16.04, 7.07, 3.40]},
    'hqc256': {'n2': 640, 'probabilities': [23.14, 34.06, 24.87, 12.02, 4.32, 1.59]}
}

def sample_vector_from_scheme(scheme):
    if scheme not in scheme_data:
        raise ValueError(f"Scheme {scheme} not recognized. Available schemes are: {list(scheme_data.keys())}.")
    
    data = scheme_data[scheme]
    n2 = data['n2']
    probabilities = np.array(data['probabilities'])
    probabilities /= 100  # Convert percentages to a proper probability sum
    probabilities /= probabilities.sum()

    # Generate the number of ones in the vector
    number_of_ones = np.random.choice(np.arange(len(probabilities)), p=probabilities)

    # Create the binary vector
    vector = np.zeros(n2, dtype=int)
    vector[:number_of_ones] = 1
    np.random.shuffle(vector)
    
    return vector

def gen_codeword(eXORu, scheme, n, n2, all_one_blocks, lib):

    cdw = np.zeros(n, dtype=int)

    # assign all_one_blocks to all one
    for i in all_one_blocks:
        start_index = i * n2
        end_index = start_index + n2
        cdw[start_index:end_index] = 1

    # setting the first block of cdw to eXORu
    start_index = 0 * n2
    end_index = start_index + n2
    cdw[start_index:end_index] = eXORu

    # convert binary bit vector to ctypes uint64
    cdw_64 = bit_array_to_uint64(lib=lib,vector=cdw,scheme=scheme)

    return cdw_64


def rm_decoder_result_wo_noise(eXORu,rm_decoder,lib,m_len,scheme):
    vector_64 = bit_array_to_uint64(lib,eXORu,scheme)
    values = [0]
    # Creating the ctypes ubyte array from the list
    message = (ctypes.c_ubyte * m_len)(*values)
    rm_decoder(message, vector_64)

    m_integer = np.frombuffer(message, dtype=np.uint8)
    return (int(m_integer[0]),)



def concatenated_decoder_result_wo_noise(scheme,lib,m_len,cdw_64):
    function_name = f"PQCLEAN_{scheme.upper()}_CLEAN_code_decode"
    PQCLEAN_CLEAN_code_decode = getattr(lib, function_name)

    PQCLEAN_CLEAN_code_decode.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # Pointer to uint8_t array (m)
        ctypes.POINTER(ctypes.c_uint64)  # Pointer to uint64_t array (em)
    ]
    PQCLEAN_CLEAN_code_decode.restype = None  # void return type

    message = (ctypes.c_uint8 * m_len)()
    
    code_word_size = len(cdw_64)
    em_array = (ctypes.c_uint64 * code_word_size)(*cdw_64)
    
    # Call the C function
    PQCLEAN_CLEAN_code_decode(message, em_array)
    return list(message)

def bernoulli_noise(rho):
    """
    Generate Bernoulli noise.
    
    Returns 0 with probability rho and 1 with probability 1 - rho.
    
    :param rho: Probability of returning 0.
    :return: int (0 or 1)
    """
    return 0 if random.random() < rho else 1