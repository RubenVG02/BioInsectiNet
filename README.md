# Fungic Bioinsecticide Discovery

## Overview

This project facilitates the discovery and design of new bioinsecticides based on target proteins. It includes tools for predicting toxicity, generating bioinsecticides, and obtaining 3D structures of designed molecules. The project leverages neural networks for toxicity prediction and bioinsecticide generation, along with genetic algorithms to refine designs.

## Usage

### Preparing Data

1. **FASTA Sequence**: Ensure your target protein is in FASTA format (amino acid sequence).

2. **Neural Networks**: You need two neural networks:
   - **Toxicity Prediction**: Use `cnn_affinity.py` to train or utilize the pre-trained model.
   - **Bioinsecticide Generation**: Use `generate_rnn.py` to train or utilize the pre-trained model.

   Alternatively, use the pre-trained models located in the `definitive_models` folder.

3. **Data**: Use data from databases such as Chembl, PubChem, or the provided "insect.csv".  


### CNN Usage (Affinity Prediction)

To predict toxicity using the CNN model, run:

```bash
python check_affinity.py --model_path <path_to_model> --data_path <path_to_data> --target_path <path_to_target_protein>
```

The program will return the toxicity of the designed bioinsecticides using the 'calculate_affinity' function.

### RNN Usage (Bioinsecticide Generation)

To generate bioinsecticides using the RNN model, run:

```bash
python pretrained_rnn.py --model_path <path_to_model> --data_path <path_to_data> --target_path <path_to_target_protein>
```

The program will return the designed bioinsecticides using the 'generate' function.

### Combination of Models

For combining both models (generation and toxicity prediction), use:

```bash
python affinity_with_target_and_generator.py --model_path <path_to_model> --data_path <path_to_data> --target_path <path_to_target_protein> --toxicity_limit <toxicity_limit> --output_path <path_to_output>
```

The program will generate bioinsecticides and filter out those exceeding the specified toxicity limit. You can also specify a path to check generated molecules.

### Genetic Algorithm

To use the genetic algorithm, run:

```bash
python genetic_algorithm.py --smiles_list <smiles_list> --csv_file <path_to_csv_file> --rnn_model <path_to_rnn_model> --model_path <path_to_model> --generations <number_of_generations> --output_path <path_to_output>
```

You can provide SMILES sequences directly, via a CSV file, or use an RNN model to guide the generation. The program will return the best SMILES sequence from the last generation.

### Installation

To obtain the 3D structure of the designed bioinsecticides, run:

```bash
python 3d_repr.py --model_path <path_to_model> --data_path <path_to_data> --target_path <path_to_target_protein> --toxicity_limit <toxicity_limit> --output_path <path_to_output>
```

This will generate an SDF file containing the 3D structure of the bioinsecticides. Use PyMOL to convert the SDF file to other formats (e.g., PDB) using the pymol_3d.py script.

## Installation

Clone the repository:

```bash
git clone https://github.com/RubenVG02/BioinsecticidesDiscovery.git
```

Or download the latest release:

```bash
wget https://github.com/RubenVG02/BioinsecticidesDiscovery/releases/latest
```

Ensure Python 3.7 or higher is installed. Install the required libraries using:

```bash
pip install -r requirements.txt
```

## Authors

- [@RubenVG02](https://www.github.com/RubenVG02)

## Features

- Design of new bioinsecticides based on the target protein
- Improving the structure of previously designed bioinsecticides based on the target protein
- Predicting the toxicity of the designed bioinsecticides
- Obtaining CSV files and screenshots of the results
- Obtaining the 3D structure of the designed bioinsecticides in different formats (SFD, PDB, etc.)
- Fast and easy to use


## Future Improvements

- Add more databases to the CNN
- Add more databases to the RNN
- More complexity to the GA

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements

- [Keras](https://keras.io/)
- [Tensorflow](https://www.tensorflow.org/)
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)






