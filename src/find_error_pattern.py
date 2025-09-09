import numpy as np
from collections import defaultdict
import copy
import logging
import sys
import os
import csv
from abc import ABC, abstractmethod
from filelock import FileLock
from util import sample_vector_from_scheme, mutate2bits, mutate_bit_by_bit, load_config, load_lib, load_rm_decoder, setup_logging, concatenated_decoder_result_wo_noise, gen_codeword, sample_binary_vector, rm_decoder_result_wo_noise


class ErrorPatternFinder(ABC):
    def __init__(self, lib, scheme, n2, n, rm_decoder, all_one_blocks, m_len):
        self.lib = lib
        self.scheme = scheme
        self.n2 = n2
        self.n = n
        self.rm_decoder = rm_decoder
        self.all_one_blocks = all_one_blocks
        self.m_len = m_len

    def calculate_shannon_entropy(self, results_distribution):
        frequencies = np.array(list(results_distribution.values()))
        total_samples = frequencies.sum()
        probabilities = frequencies / total_samples
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))

    @abstractmethod
    def evaluate_error_pattern(self, error_pattern, num_iter):
        """Evaluate the error pattern. Must be implemented by subclasses."""
        pass

    def write_results_to_csv(self, filepath, vectors, entropy):
        lockfile = filepath + '.lock'
        file_exists = os.path.exists(filepath)

        with FileLock(lockfile):
            with open(filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                # If file is new, write headers
                if not file_exists:
                    writer.writerow(['Vectors', 'Entropy'])
                writer.writerow([vectors, entropy])

    def find_optimal_error_pattern(self, num_generation, num_children, output_filepath):
        
        # generate the starting e vector
        if self.scheme == 'hqc128':
            e_0 = sample_binary_vector(self.n2, 0.42, 0.44)
        else:
            e_0 = sample_binary_vector(self.n2, 0.435, 0.460)

        shannon_entropy_0, results_distribution_0 = self.evaluate_error_pattern(error_pattern=e_0, num_iter=2000)
        logging.info("Starting vector Shannon entropy: {}".format(shannon_entropy_0))
        logging.info("Results: {}".format(results_distribution_0))

        generation_best_shannon_entropy = shannon_entropy_0
        generation_best_vectors = copy.deepcopy(e_0)
        generation_best_results_distribution = results_distribution_0

        for generation in range(num_generation):
            children_best_shannon_entropy = generation_best_shannon_entropy
            children_best_vectors = copy.deepcopy(generation_best_vectors)
            children_best_results_distribution = generation_best_results_distribution

            for _ in range(num_children):
                if generation_best_shannon_entropy < 2.5:
                    new_mutation = mutate2bits(copy.deepcopy(generation_best_vectors))
                    num_iter = 2000
                else:
                    new_mutation = mutate_bit_by_bit(copy.deepcopy(generation_best_vectors))
                    num_iter = 5000

                new_mutation_shannon_entropy, new_mutation_results_distribution = self.evaluate_error_pattern(error_pattern=new_mutation, num_iter=num_iter)

                if new_mutation_shannon_entropy > children_best_shannon_entropy:
                    children_best_shannon_entropy = new_mutation_shannon_entropy
                    children_best_vectors = new_mutation
                    children_best_results_distribution = new_mutation_results_distribution

            if children_best_shannon_entropy > generation_best_shannon_entropy:
                generation_best_vectors = children_best_vectors
                generation_best_shannon_entropy = children_best_shannon_entropy
                generation_best_results_distribution = children_best_results_distribution
                logging.info("generation {} best Shannon entropy: {}".format(generation, generation_best_shannon_entropy))
                logging.info("Current best results distribution: {}".format(generation_best_results_distribution))

            #if generation==20 and generation_best_shannon_entropy < 1:
            #    print("Shannon entropy smaller than 1 after 20 generations. Exiting...")
            #    sys.exit(10)
        
        logging.info("Optimized vector Shannon entropy: {}".format(generation_best_shannon_entropy))
        self.write_results_to_csv(output_filepath, generation_best_vectors, generation_best_shannon_entropy)

# Subclass for MVFinder
class MVFinder(ErrorPatternFinder):
    def evaluate_error_pattern(self, error_pattern, num_iter):
        results_distribution = defaultdict(int)
        for _ in range(num_iter):
            u = sample_vector_from_scheme(self.scheme)
            # u XOR error pattern
            vector_sum = np.bitwise_xor(u, error_pattern)
            # generate codeword from vector sum
            cdw_64 = gen_codeword(eXORu=vector_sum, scheme=self.scheme, n=self.n, n2=self.n2, all_one_blocks=self.all_one_blocks, lib=self.lib)
            # result from the concatenated decoder
            concatenated_decoder_result_tmp = concatenated_decoder_result_wo_noise(scheme=self.scheme,lib=self.lib,m_len=self.m_len,cdw_64=cdw_64)
            # turn the result into a tuple and add to the count of the result
            results_distribution[tuple(concatenated_decoder_result_tmp)] += 1
        shannon_entropy = self.calculate_shannon_entropy(results_distribution)
        return shannon_entropy, results_distribution
    
# Subclass for FDFinder
class FDFinder(ErrorPatternFinder):
    def evaluate_error_pattern(self, error_pattern, num_iter):
        results_distribution = defaultdict(int)
        for _ in range(num_iter):
            u = sample_vector_from_scheme(self.scheme)
            # u XOR error pattern
            vector_sum = np.bitwise_xor(u, error_pattern)
            # result from the rm decoder
            rm_decoder_result_tmp = rm_decoder_result_wo_noise(eXORu=vector_sum,rm_decoder=self.rm_decoder,lib=self.lib,m_len=self.m_len,scheme=self.scheme)
            # turn the result into a tuple and add to the count of the result
            results_distribution[rm_decoder_result_tmp] += 1
        shannon_entropy = self.calculate_shannon_entropy(results_distribution)
        return shannon_entropy, results_distribution


def create_finder(mvORfd, lib, scheme, n, n2, all_one_blocks, rm_decoder, m_len):
    if mvORfd == 'mv':
        return MVFinder(lib=lib, scheme=scheme, n=n, n2=n2, all_one_blocks=all_one_blocks, rm_decoder=rm_decoder, m_len=m_len)
    elif mvORfd == 'fd':
        return FDFinder(lib=lib, scheme=scheme, n=n, n2=n2, all_one_blocks=all_one_blocks, rm_decoder=rm_decoder, m_len=m_len)
    else:
        raise ValueError("Invalid type. Use 'mv' or 'fd'.")
    
def main(scheme,mvORfd,output_filepath):
    # Load configuration and setup
    params = load_config(scheme)
    lib_path = params.get("lib path", "lib path_not_specified")
    lib = load_lib(lib_path=lib_path)
    rm_decoder = load_rm_decoder(lib=lib)

    n2 = params.get("n2", "n2_not_specified")
    n = params.get("n", "n_not_specified")
    m_len = params.get("message length", "message length_not_specified")
    all_one_blocks = params.get("all one blocks", [])

    # Setup logging
    setup_logging(script="find_error_pattern", scheme=scheme, mvORfd = mvORfd)

    error_pattern_finder = create_finder(mvORfd, lib=lib, scheme=scheme, n=n, n2=n2, all_one_blocks=all_one_blocks, rm_decoder=rm_decoder,m_len=m_len)
    
    # Start finding the optimal error pattern
    error_pattern_finder.find_optimal_error_pattern(num_generation=120, num_children=30, output_filepath=output_filepath)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <scheme><mvORfd><output_filepath>")
        sys.exit(1)
    
    main(scheme=sys.argv[1], mvORfd=sys.argv[2], output_filepath=sys.argv[3])
