# BioInsectiNet: Neural Network and Genetic Algorithm Framework for Bioinsecticide Design

## Overview

**BioInsectiNet** is an advanced computational pipeline designed to accelerate the discovery and optimization of bioinsecticides targeting specific proteins. By combining deep learning models—including Recurrent Neural Networks (RNNs) for de novo molecule generation and Transformer-based models for affinity and toxicity prediction—with genetic algorithms for iterative molecular optimization, the project enables precision bioinsecticide design.

The framework also integrates molecular docking tools to analyze 3D interactions between optimized compounds and their protein targets, facilitating a comprehensive in silico workflow from molecule generation to interaction validation. BioInsectiNet leverages publicly available chemical and biological databases (e.g., ChEMBL) for training and evaluation, making it adaptable and extensible for various insecticide design projects.

## Usage

Preparing Data for Model Training

* **For Affinity Prediction Models (e.g., Transformer-based models):**
Prepare CSV files containing SMILES, FASTA sequences, and associated IC50 values. Public databases like ChEMBL can be used as data sources.
Use the script to preprocess data automatically:
```bash
python prepare_data.py --input_csv <path_to_input_csv> --output_dir <path_to_output_dir>
``` 

* **For RNN-based Molecule Generation Models:**
Prepare a plain text or .smi file with SMILES strings separated by newline characters. These can be sourced from chemical databases or generated experimentally.


### Affinity and Toxicity Prediction

Predict the binding affinity and toxicity of candidate bioinsecticides against a target protein using pre-trained Transformer models.

- Pre-trained models are available in `models/checkpoints`.
- To predict the binding affinity and toxicity of a designed bioinsecticide, run:


```bash
python check_affinity.py --model_path <path_to_model> --data_path <path_to_data> --target_path <path_to_target_protein> ...
```

This script outputs predicted affinity and toxicity scores using the internal `calculate_affinity` function.

### Bioinsecticide Generation with RNN Models

Generate novel bioinsecticide candidates using Recurrent Neural Networks trained on SMILES data.

- Pre-trained RNN models are stored in `models/generator`.
- Review the README in `models/generator` to select the model that best fits your study needs.
- To generate new bioinsecticides, run:


```bash
python generation_RNN.py --model_path <path_model> --data_path <path_data_model> --save_dir <save_dir> --num_molecules <num_generated_molecules> ...

```

This will generate new SMILES and optionally save visualizations of the molecules.

### Integrated Generation and Affinity Filtering

Automatically generate bioinsecticide candidates and filter them by predicted affinity/toxicity thresholds in a single pipeline.

```bash
python affinity_with_target_and_generator.py --model_path <path_to_model> --data_path <path_to_data> --target_path <path_to_target_protein> --toxicity_limit <toxicity_limit> --output_path <path_to_output> ...
```

This workflow produces candidate molecules optimized to meet your target IC50 (toxicity) constraints and supports uploading results to cloud storage.

### Genetic Algorithm Optimization

Optimize candidate molecules starting either from generated SMILES or from an existing dataset to improve binding affinity and reduce toxicity.

- You can provide SMILES lists directly, CSV files, or generate them on-the-fly via the RNN generator.

- To run the genetic algorithm optimization, use:


```bash
python src/genetic_algorithm.py --generate_smiles_from_optimizer --target_fasta <path_to_target_fasta> --objective_ic50 <objective_ic50_value> --generations <num_generations> --population_size <population_size> --num_parents_select <num_parents_select> --mutation_rate <mutation_rate> --stagnation_limit <stagnation_limit> --output_dir <output_dir> --all_results_file <all_results_file> --best_overall_file <best_overall_file> --initial_best_file <initial_best_file> --image_dir <image_dir> ...
```

The GA will iteratively improve molecule sets over the specified generations, selecting and mutating molecules to optimize affinity and other metrics. In this case, as we use the --generate_smiles_from_optimizer flag, the program will generate new SMILES sequences based on the target protein and the objective IC50 value using the combined RNN and Transformer model approach.



### Molecular Docking and 3D Interaction Analysis

Analyze the 3D binding interactions of optimized bioinsecticide candidates with their protein targets using docking.

Run docking with the following command:

```bash
python docking_pipeline.py --smile "<SMILES_string>" --fasta "<FASTA_sequence>" --center <center_coordinates> --box_size <box_size> --output_dir_docking <output_dir_docking> --vina_path <path_to_vina>

```

This step generates 3D ligand-receptor complexes for visual analysis and further computational studies. This scripts outputs PDBQT files for both the receptor and ligand, which can be used in molecular visualization tools like PyMOL or Chimera.



## Installation

Clone the repository:

```bash
git clone https://github.com/RubenVG02/BioInsectiNet.git
cd BioInsectiNet
```

Or download the latest release:

```bash
wget https://github.com/RubenVG02/BioInsectiNet/releases/latest
```

## Setup environment

Create and activate the Conda environment with all dependencies using the provided environment.yml:

```bash
conda env create -f environment.yml
conda activate BioInsectiNet
```


## Authors

- [@RubenVG02](https://www.github.com/RubenVG02)

## Features

- De novo generation of bioinsecticide candidates using advanced RNN models.

- Affinity and toxicity prediction with Transformer architectures for precise evaluation.

- Genetic algorithm-driven optimization of molecule sets for improved bioactivity and safety profiles.

- Automated molecular docking pipeline integrating RDKit, Meeko, and AutoDock Vina for 3D interaction analysis.

- Support for multi-format 3D molecular representations (SDF, PDB).

- Data preprocessing tools compatible with public databases like ChEMBL for seamless training data preparation.

- Modular and scalable pipeline adaptable for drug discovery, agrochemical design, and computational chemistry applications.

- Comprehensive logging, visualization, and result export in CSV and image formats.

## Future Improvements

- Integration with additional chemical and biological databases (PubChem, ZINC, BindingDB) for richer training datasets.

- Implementation of Transformer-based molecule generation models alongside RNNs for enhanced chemical space exploration.

- Improved cloud-based storage and distributed computing support for large-scale screening.

- Incorporation of ADMET prediction modules for holistic drug-like property assessment.

- Enhanced user interface (web or GUI) for easier pipeline interaction by non-programmers.

- Expansion to multi-target bioinsecticide design via multi-objective optimization.

- Deployment of active learning loops to iteratively refine models with experimental feedback.


## License

This project is licensed under the MIT License. 
