import numpy as np
from collections import defaultdict
import sys
import pickle
import pandas as pd
from abc import ABC, abstractmethod
from util import sample_vector_from_scheme, load_config, load_lib, load_rm_decoder, setup_logging, concatenated_decoder_result_wo_noise, gen_codeword, rm_decoder_result_wo_noise


class TemplateCreator(ABC):
    def __init__(self, lib, scheme, n2, n, rm_decoder, all_one_blocks, m_len, config):
        self.lib = lib
        self.scheme = scheme
        self.n2 = n2
        self.n = n
        self.rm_decoder = rm_decoder
        self.all_one_blocks = all_one_blocks
        self.m_len = m_len
        self.config = config
    
    def select_error_patterns(self, error_patterns):

        # Method 1: Use output directly from find_error_patterns.py
        # Filter and sort error patterns based on entropy threshold
        '''
        entropy_threshold = self.config['entropy_threshold']
        selected_error_patterns = np.stack(error_patterns[error_patterns['Entropy'] > entropy_threshold].sort_values(by='Entropy', ascending=False)['Vectors'].to_numpy())
        # Save the filtered patterns with the filename from config
        output_file = self.config['output_file']
#        print(selected_error_patterns.shape,np.array(selected_error_patterns[0]))
        np.save(output_file, selected_error_patterns)
        
        '''

        # Method 2: Use the same error patterns as in the paper
        output_file = self.config['output_file']
        selected_error_patterns = np.load(output_file)
        

        return selected_error_patterns
        
    @abstractmethod
    def decode_result(self, vector_sum):
        # Abstract method to decode the result based on the oracle type (MV or FD).
        pass
    
    def save_probability_template_to_pickle(self, probability_template, error_pattern_index):
        template_path = self.config['template_path']
        with open(f'{template_path}/error_pattern_{error_pattern_index}.pkl', 'wb') as file:
            pickle.dump(probability_template, file)

    def create_template(self, error_patterns):
        selected_patterns = self.select_error_patterns(error_patterns)
        num_samples = self.config['num_samples']
        
        for i in range(len(selected_patterns)): 
            error_pattern = np.array(selected_patterns[i])
            results_distribution = defaultdict(int)
            u_aggregate = defaultdict(lambda: np.zeros(self.n2, dtype=int))

            for _ in range(num_samples):
                u = sample_vector_from_scheme(self.scheme)
                vector_sum = np.bitwise_xor(u, error_pattern)
                decoded_result = self.decode_result(vector_sum)
                decoded_result = tuple(decoded_result)
                u_aggregate[decoded_result] += u
                results_distribution[decoded_result] += 1

            # Calculate probability template
            probability_template = {}
            total_samples = sum(results_distribution.values())

            for decoder_result in list(u_aggregate.keys()):
                decoder_result_percentage = (results_distribution[decoder_result] / total_samples) * 100
                print(f'Error pattern {i}: decoder result {decoder_result}, percentage {decoder_result_percentage:.2f}%.')
                probability_template[decoder_result] = u_aggregate[decoder_result] / results_distribution[decoder_result]
                # Filter out results not happening once out of 2000 times
                if decoder_result_percentage<1/2000*100:
                    del u_aggregate[decoder_result]
                    del results_distribution[decoder_result]
                    del probability_template[decoder_result]


            # Save template
            self.save_probability_template_to_pickle(probability_template, i)


class MVTemplateCreator(TemplateCreator):
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

class FDTemplateCreator(TemplateCreator):
    def decode_result(self, vector_sum):
        return rm_decoder_result_wo_noise(
            eXORu=vector_sum,
            rm_decoder=self.rm_decoder,
            lib=self.lib,
            m_len=self.m_len,
            scheme=self.scheme
        )


def get_template_creator(mvORfd, **kwargs):
    # Dynamically return the appropriate creator
    if mvORfd == 'mv':
        return MVTemplateCreator(**kwargs)
    elif mvORfd == 'fd':
        return FDTemplateCreator(**kwargs)


def main(scheme,mvORfd):
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
    setup_logging(script="create_template", scheme=scheme, mvORfd = mvORfd)

    # load the optimized error patterns
    error_patterns = pd.read_csv(f"./output/{mvORfd}_error_patterns_{scheme}.csv")
    # Parse the 'Vectors' column into NumPy arrays
    error_patterns['Vectors'] = error_patterns['Vectors'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ").astype(int) if isinstance(x, str) else np.array(x, dtype=int))

    # different configurations for mv and fd.
    config = {
        'mv': {
            'entropy_threshold': 1.9,
            'num_samples': 5000000,
            'output_file': f"./data_input/mv_selected_error_patterns_{scheme}.npy",
            'template_path': f"./template/mv/{scheme}"
        },
        'fd': {
            'entropy_threshold': 2.8,
            'num_samples': 10000000,
            'output_file': f"./data_input/fd_selected_error_patterns_{scheme}.npy",
            'template_path': f"./template/fd/{scheme}/"
        }
    }

    # validate the mvORfd input
    if mvORfd not in config:
        raise ValueError(f"Invalid mvORfd value: {mvORfd}. Supported values are 'mv' and 'fd'.")

    # Call the template creator
    template_creator = get_template_creator(
        mvORfd=mvORfd,
        lib=lib,
        scheme=scheme,
        config=config[mvORfd],
        rm_decoder=rm_decoder,
        n2=n2,
        n=n,
        all_one_blocks=all_one_blocks,
        m_len=m_len
    )

    # Create templates
    template_creator.create_template(error_patterns)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <scheme><mvORfd>")
        sys.exit(1)
    
    main(scheme=sys.argv[1], mvORfd=sys.argv[2])




