import numpy as np
import pickle
import logging
import sys
from scipy.spatial.distance import hamming
from itertools import combinations
from abc import ABC, abstractmethod
import random
import math
from util import load_config, load_lib, load_rm_decoder, concatenated_decoder_result_wo_noise, gen_codeword, rm_decoder_result_wo_noise, bernoulli_noise


class AttackSimulator(ABC):
    def __init__(self, lib, scheme, n2, n, n1, w, rm_decoder, all_one_blocks, m_len, use_error_patterns, templates,rho):
        self.lib = lib
        self.scheme = scheme
        self.n2 = n2
        self.n = n
        self.n1 = n1
        self.w = w
        self.rm_decoder = rm_decoder
        self.all_one_blocks = all_one_blocks
        self.m_len = m_len
        self.use_error_patterns = use_error_patterns
        self.templates = templates
        self.rho = rho

        # parameter l for the simulated attack
        if self.scheme == 'hqc128':
            self.l = 51
        else:
            self.l = 61

    @abstractmethod
    def decode_result(self, vector_sum):
        """Subclasses must implement this method to decode results."""
        pass

    def generate_full_vector_dropping_excess(self):
        vector = np.zeros(self.n, dtype=int)
        vector[:self.w] = 1
        np.random.shuffle(vector)
        return vector[:self.n1 * self.n2]
    
    def generate_probability_blocks(self, vector):
        vector_blocks = np.array_split(vector, int(len(vector)/self.n2))
        vector_probability_blocks = np.empty_like(vector_blocks, dtype=float)

        for i, vector_block in enumerate(vector_blocks):
            block_probability = np.ones(self.n2)
            for j, error_pattern in enumerate(self.use_error_patterns):
                # Decode the result using the mvORfd-specific logic
                vector_sum = np.bitwise_xor(vector_block, error_pattern)

                mtmp = self.decode_result(vector_sum)
                mtmp = tuple(mtmp)
                
                # Handle noise and missing keys
                if bernoulli_noise(self.rho) != 0:
                    mtmp = 999
                template = self.templates[j]
                if mtmp not in template:
                    mtmp = random.choice(list(template.keys()))
                probability = np.array(template[mtmp])
                block_probability *= probability
            vector_probability_blocks[i] = block_probability

        return vector_probability_blocks.ravel()
    
    def sample_and_sort_according_to_probability(self):
        x = self.generate_full_vector_dropping_excess()
        y = self.generate_full_vector_dropping_excess()
        xy = np.concatenate((x, y))
        xy_probabilities = self.generate_probability_blocks(vector=xy)
        sorted_indices = np.argsort(xy_probabilities)
        sorted_xy = xy[sorted_indices]
        return sorted_xy
    
    def calculate_num_errors_half_length(self,array):
        half_sorted_xy = array[:self.n]
        return sum(half_sorted_xy)

    def get_half_sorted_xy_plus_l(self, array):
        half_sorted_xy_plus_l = array[:self.n+self.l]
        return half_sorted_xy_plus_l

    def exclude_random_positions(self, array, T1, T0):
        
        if T1 >= T0:
            raise ValueError("T0 must be less than T1")
    
        new_array = array[:T0]
        # Generate all indices from 0 to T0-1
        all_indices = np.arange(T0)
        
        # Randomly choose T1 indices to exclude
        exclude_indices = np.random.choice(all_indices, T1, replace=False)
        
        # Get the mask of indices to include by checking which are not in exclude_indices
        include_mask = np.isin(all_indices, exclude_indices, invert=True)
        
        # Use the mask to select the remaining elements
        remaining_array =new_array[include_mask]
    
        excluded_array = new_array[exclude_indices]

        return remaining_array, excluded_array
    
    def random_split_array(self, array, array_length):
        # Check that the number of elements is even
        if len(array) % 2 != 0:
            raise ValueError("The array must contain an even number of elements")
        
        indices = np.arange(array_length)  # Create an array of indices
        np.random.shuffle(indices)  # Shuffle the indices

        # Split indices for preserving order within each split
        first_half_indices = sorted(indices[:array_length//2])
        second_half_indices = sorted(indices[array_length//2:])

        # Use these indices to form the new arrays
        first_half = array[first_half_indices]
        second_half = array[second_half_indices]

        return first_half, second_half
    

    def calculate_success_rate(self, half_sorted_xy_plus_l, maxW, T0, T1, T):

        half_sorted_xy_plus_l_weight = half_sorted_xy_plus_l.sum()

        # if the weight is larger than maxW, the split will end up with more than maxW/2 errors in one list, and cannot be solved. --> failure 
        if half_sorted_xy_plus_l_weight > maxW:
            return [half_sorted_xy_plus_l_weight,float('inf')]

        found = False

        for j in range(T):
            
            T0_minus_T1, excluded_array = self.exclude_random_positions(array=half_sorted_xy_plus_l, T1=T1, T0=T0)

            if sum(excluded_array)>0: # excluded array contains one. draw another round.
                num_draw=j+1
                continue
            
            L=np.concatenate((T0_minus_T1,half_sorted_xy_plus_l[T0:]))

            lenL = len(L)

            L1, L2 = self.random_split_array(array=L, array_length=lenL)

            # if any list is greater than maxW/2, cannot be solved, draw another round 
            if sum(L1)<=maxW/2 and sum(L2)<=maxW/2:
                num_draw=j+1
                found=True
                break

        if not found:
            num_draw=float('inf')
        
        return [half_sorted_xy_plus_l_weight, num_draw]


    def simulate_attack(self):
        # Perform attack simulation
        error_stats = []
        num_samples = 10000

        result = []
        for _ in range(num_samples):
            # sample xy and sort according to probability
            sorted_xy = self.sample_and_sort_according_to_probability()

            # Part 1: calculate number of errors at half-length for each sample
            num_errors_half_length = self.calculate_num_errors_half_length(array=sorted_xy)
            error_stats.append(num_errors_half_length)

            # Part 2: evaluate success or not
            # parameters
            maxW = 4
            if self.scheme == 'hqc128':
                T0=12000
                T1=10000
            elif self.scheme =='hqc192':
                T0=24000
                T1=18000
            elif self.scheme == 'hqc256':
                T0=40000
                T1=30000        
            T=16 # number of random draws and splits 
            half_sorted_xy_plus_l = self.get_half_sorted_xy_plus_l(sorted_xy)

            success_trial =self.calculate_success_rate(half_sorted_xy_plus_l=half_sorted_xy_plus_l, maxW=maxW, T0=T0, T1=T1, T=T)
            result.append(success_trial)

        result_list = [t[1] for t in result] 
        success_rates = sum(1 for x in result_list if math.isfinite(x))/len(result_list)


        logging.info(f'maxW={maxW} success rate: {success_rates}')
        logging.info(f'result with rho={self.rho}, T0={T0}, T1={T1}, T={T}, maxW={maxW}: {result}')
                    
        
        return error_stats, success_rates

class MVAttackSimulator(AttackSimulator):
    def decode_result(self, vector_sum):
        # Step 1: Generate codeword
        cdw_64 = gen_codeword(
            eXORu=vector_sum,
            scheme=self.scheme,
            n=self.n,
            n2=self.n2,
            all_one_blocks=self.all_one_blocks,
            lib=self.lib
        )
        # Step 2: Decode result using concatenated decoder
        return concatenated_decoder_result_wo_noise(
            scheme=self.scheme,
            lib=self.lib,
            m_len=self.m_len,
            cdw_64=cdw_64
        )

class FDAttackSimulator(AttackSimulator):
    def decode_result(self, vector_sum):
        return rm_decoder_result_wo_noise(
            eXORu=vector_sum,
            rm_decoder=self.rm_decoder,
            lib=self.lib,
            m_len=self.m_len,
            scheme=self.scheme
        )

def load_template(mvORfd, scheme, num_error_patterns):
    error_patterns = np.load(f"./data_input/{mvORfd}_selected_error_patterns_{scheme}.npy")
    template_path = f"./template/{mvORfd}/{scheme}"

    # Select error patterns based on entropy (if only one error pattern) or hamming distance (if more than one)
    use_error_patterns = []
    if num_error_patterns == 1:
        use_error_patterns_index = [0] # the first error pattern has the larget entropy
    else:
        indices_combinations = combinations(range(error_patterns.shape[0]), num_error_patterns)
        best_indices = max(indices_combinations, key=lambda indices: sum(
            hamming(error_patterns[i], error_patterns[j]) for i, j in combinations(indices, 2)
        ))
        use_error_patterns_index = list(best_indices)
    logging.info("The error patterns to use has index {}".format(use_error_patterns_index))

    use_error_patterns = []
    templates = []
    for i in use_error_patterns_index:
        use_error_patterns.append(error_patterns[i])
        with open(f'{template_path}/error_pattern_{i}.pkl', 'rb') as file:
            template = pickle.load(file)
        templates.append(template)

    return use_error_patterns, templates

def save_stats(scheme, mvORfd, num_error_patterns,rho, error_stats):
    # Save results
    output_file = f'./output/{mvORfd}_num_error_patterns={num_error_patterns}_rho={rho}_{scheme}.npy'
    np.save(output_file, np.array(error_stats))
    mean_errors = np.mean(error_stats)
    logging.info(f"{mvORfd} oracle for {scheme} with {num_error_patterns} chosen error patterns. Expected number of errors: {mean_errors}.")

def setup_logging(scheme):
    # Constants
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    # Remove all handlers associated with the root logger (clean up)
    DATE_FORMAT = '%m/%d/%Y %H:%M:%S'
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    # Set up new logging configuration
    logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT, handlers=[
        logging.FileHandler(f'./logs/simulate_attack_{scheme}.log'),
        logging.StreamHandler()
    ])

def main(scheme, num_error_patterns, rho, mvORfd):
    # Load configuration and setup
    params = load_config(scheme)
    lib_path = params.get("lib path", "lib path_not_specified")
    lib = load_lib(lib_path=lib_path)
    rm_decoder = load_rm_decoder(lib=lib)

    n2 = params.get("n2", "n2_not_specified")
    n = params.get("n", "n_not_specified")
    m_len = params.get("message length", "message length_not_specified")
    all_one_blocks = params.get("all one blocks", [])
    n1 = params.get("n1", "n1_not_specified")
    w = params.get("w", "w_not_specified")

    setup_logging(scheme)

    # load error patterns and corresponding templates
    use_error_patterns, templates = load_template(mvORfd, scheme, num_error_patterns)
    

    # Choose the appropriate simulator
    if mvORfd == 'mv':
        simulator = MVAttackSimulator(
            lib=lib,
            scheme=scheme,
            n2=n2,
            n=n,
            n1=n1,
            w=w,
            rm_decoder=rm_decoder,
            all_one_blocks=all_one_blocks,
            m_len=m_len,
            use_error_patterns=use_error_patterns,
            templates=templates,
            rho=rho
        )
    elif mvORfd == 'fd':
        simulator = FDAttackSimulator(
            lib=lib,
            scheme=scheme,
            n2=n2,
            n=n,
            n1=n1,
            w=w,
            rm_decoder=rm_decoder,
            all_one_blocks=all_one_blocks,
            m_len=m_len,
            use_error_patterns=use_error_patterns,
            templates=templates,
            rho=rho
        )
    else:
        raise ValueError(f"Invalid mvORfd: {mvORfd}. Must be 'mv' or 'fd'.")

    # Run simulation
    error_stats, success_rates = simulator.simulate_attack()

    # Save statistics
    save_stats(scheme, mvORfd, num_error_patterns, rho, error_stats)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python script.py <scheme> <mvORfd> <rho> <num_error_patterns> ")
        sys.exit(1)
    main(scheme=sys.argv[1], mvORfd=sys.argv[2], rho=float(sys.argv[3]), num_error_patterns=int(sys.argv[4]))
