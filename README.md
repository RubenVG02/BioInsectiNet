# BioInsectiNet: Neural Network and Genetic Algorithm Framework for Bioinsecticide Design

## Overview

**BioInsectiNet** is an advanced computational pipeline designed to accelerate the discovery and optimization of bioinsecticides targeting specific proteins. By combining deep learning models—including Recurrent Neural Networks (RNNs) for de novo molecule generation and Transformer-based models for affinity and toxicity prediction—with genetic algorithms for iterative molecular optimization, the project enables precision bioinsecticide design.

The framework also integrates molecular docking tools to analyze 3D interactions between optimized compounds and their protein targets, facilitating a comprehensive in silico workflow from molecule generation to interaction validation. BioInsectiNet leverages publicly available chemical and biological databases (e.g., ChEMBL) for training and evaluation, making it adaptable and extensible for various insecticide design projects.

## Main Pipeline Usage

The `main.py` script provides a comprehensive, automated pipeline for bioinsecticide design that integrates all components of BioInsectiNet. This is the recommended way to use the framework.

### Quick Start

Run the complete pipeline with default settings:

```bash
python main.py --target_fasta "MKVLWAALLVTFLAGCQA..." --run_all
```

### Pipeline Components

The main pipeline consists of four sequential steps that can be run independently or together:

1. **Molecule Generation**: Generate novel bioinsecticide candidates using RNN models
2. **Affinity Filtering**: Filter molecules based on predicted binding affinity to target protein
3. **Genetic Algorithm Optimization**: Optimize molecules using evolutionary approaches
4. **Molecular Docking**: Perform 3D interaction analysis with target protein

### Pipeline Options

**Run Complete Pipeline:**

```bash
python main.py --target_fasta "PROTEIN_SEQUENCE" --run_all
```

**Run Specific Steps:**

```bash
# Generation and affinity filtering only
python main.py --target_fasta "PROTEIN_SEQUENCE" --run_generation --run_affinity

# Genetic algorithm optimization on existing molecules
python main.py --target_fasta "PROTEIN_SEQUENCE" --input_smiles_file molecules.smi --run_ga

# Docking analysis with top 5 molecules
python main.py --target_fasta "PROTEIN_SEQUENCE" --run_generation --run_affinity --run_docking --num_molecules_docking 5
```

### Key Parameters

- `--target_fasta`: Target protein sequence (required)
- `--total_generated`: Number of molecules to generate (default: 500)
- `--accepted_value`: IC50 threshold for filtering (default: 1000.0)
- `--objective_ic50`: Target IC50 for optimization (default: 50.0)
- `--generations`: GA generations (default: 100)
- `--num_molecules_docking`: Molecules to dock (default: 3)

### Output Structure

Each pipeline run creates a timestamped results directory:

```
bioinsectinet_results_YYYYMMDD_HHMMSS/
├── 01_generation/           # Generated molecules
├── 02_affinity_filtering/   # Filtered candidates
├── 03_genetic_algorithm/    # Optimized molecules
├── 04_docking/             # Docking results and 3D structures
├── pipeline_log_*.txt      # Execution log
├── session_config.json     # Configuration backup
└── PIPELINE_REPORT.md      # Comprehensive results summary
```

## Individual Component Usage

For advanced users who want to run specific components independently:

### Data Preparation

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

This step generates 3D ligand-receptor complexes for visual analysis and further computational studies. This script outputs PDBQT files for both the receptor and ligand, which can be used in molecular visualization tools like PyMOL or Chimera.

### Advanced Pipeline Examples

**Custom Generation Parameters:**

```bash
python main.py --target_fasta "PROTEIN_SEQUENCE" --run_all \
  --total_generated 2000 \
  --generations 100 \
  --accepted_value 1000.0 \
  --min_length 15 \
  --max_length 200
```

**High-Throughput Screening:**

```bash
python main.py --target_fasta "PROTEIN_SEQUENCE" --run_all \
  --total_generated 5000 \
  --max_molecules 100 \
  --generations 200 \
  --num_molecules_docking 10
```

**Production Run with Visualization:**

```bash
python main.py --target_fasta "PROTEIN_SEQUENCE" --run_all \
  --draw_lowest \
  --smiles_to_draw 5 \
  --generate_qr \
  --upload_to_mega
```

### Pipeline Results Interpretation

After completion, the main pipeline generates:

- **Pipeline Report**: Comprehensive markdown report with all results
- **Log Files**: Detailed execution logs for debugging and analysis
- **Molecule Images**: 2D visualizations of top candidates
- **Docking Structures**: 3D PyMOL sessions for interactive analysis
- **CSV Files**: Structured data for further analysis
- **Configuration Backup**: Complete parameter settings for reproducibility

## Installation

Clone the repository:

```bash
git clone https://github.com/RubenVG02/BioInsectiNet.git
cd BioInsectiNet
```

### Quick Setup

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate BioInsectiNet
```

**Note**: AutoDock Vina will be automatically downloaded if not found in the system PATH.

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
