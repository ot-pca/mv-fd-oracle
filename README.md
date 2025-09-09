
This is the repo for the paper **Multi-Value Plaintext-Checking and Full-Decryption Oracle-Based Attacks on HQC from Offline Templates**.

Link to paper:

### Overview

The source code for finding patterns, creating templates, and simulating attacks is located in the `src/` folder. The error patterns found and used for each scheme and oracle attack in the paper are located in the `data_input/` folder.

### Folder Structure

```
main folder/
├── bin/
│   └── sh files for running the source code in src/
├── data_input/
│   └── npy files containing selected error patterns
├── hqc128/
│   └── clean/
├── logs/
├── output/
├── plots/
├── src/
│   ├── find_error_pattern.py  # Script to optimize error patterns
│   ├── create_template.py     # Script to generate templates
│   ├── simulate_attack.py     # Script to simulate attacks
│   └── util.py                # Utility functions for the pipeline
├── template/
│   ├── fd/hqc128/
│   └── mv/hqc128/
└── config.json                # Configuration file for schemes  
```

### Note

The `hqc128/clean` directory contains the standard PQClean implementation of hqc-128 from [PQClean](https://github.com/PQClean/PQClean), with a single modification made by the paper's authors to facilitate their analysis.

A custom function, `reed_muller_decode_one_block`, was added to the `hqc128/clean/reed_muller.c` file. Its purpose is to retrieve the decoding output for a single Reed-Muller block, which is used for the offline error pattern finding and template construction of the FD attack.