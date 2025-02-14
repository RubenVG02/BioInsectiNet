# Molecule Generator Models

This directory contains RNN-based models with LSTM layers trained to generate molecular structures represented as SMILES strings. Each model has been trained on a different dataset, with variations in the size and type of SMILES strings used. Below is a detailed description of each model.

---

## Models Overview

### 1. **Model: `bindingDB_smiles_filtered_v1.pth`**

- **Dataset**: BindingDB SMILES filtered by size.
- **Description**: This model was trained on SMILES strings from the BindingDB database, filtered to include only those with fewer than 150 elements. The filtering was done to significantly reduce computational cost and improve generation efficiency without significantly affecting the quality of the generated molecules.
- **Training Details**:
  - Epochs: 50
  - Time per epoch: 15 minutes
- **Use Case**: Ideal for generating smaller molecules efficiently.

---

### 2. **Model: `bindingDB_smiles_longest_v2.pth`**

- **Dataset**: BindingDB SMILES with longer sequences.
- **Description**: This model was trained on SMILES strings from the BindingDB database, specifically including only those with more than 150 elements. It is designed to generate larger and more complex molecular structures.
- **Training Details**:
  - Epochs: 50
  - Time per epoch: 2 minutes
- **Use Case**: Suitable for generating larger and more complex molecules. However, this model requires significantly more time for generation compared to `bindingDB_smiles_filtered_v1.pth`.

---

### 3. **Model: `chembl_smiles_filtered_v1.pth`**

- **Dataset**: ChEMBL SMILES filtered by size.
- **Description**: This model was trained on SMILES strings from the ChEMBL database, filtered to include only those with fewer than 150 elements. The filtering reduces computational cost while maintaining generation effectiveness.
- **Training Details**:
  - Epochs: 50
  - Time per epoch: 10 minutes
- **Use Case**: Efficient generation of smaller and medium molecules.

---

### 4. **Model: `chembl_smiles_longest_v1.pth`**

- **Dataset**: ChEMBL SMILES with longer sequences.
- **Description**: This model was trained on SMILES strings from the ChEMBL database, specifically including only those with more than 150 elements. It is designed to generate larger and more complex molecular structures.
- **Training Details**:
  - Epochs: 50
  - Time per epoch: 5 minutes
- **Use Case**: Suitable for generating larger molecules. As with the other `longest' models, generation time is significantly high.

---

### 5. **Model: `GDBMedChem_v1.pth`**

- **Dataset**: GDBMedChem database.
- **Description**: This model was trained on SMILES strings from the GDBMedChem database, which contains a diverse set of small, drug-like molecules. The dataset is known for its coverage of chemical space, making this model well-suited for generating novel, synthetically accessible molecules with potential drug-like properties.
- **Training Details**:
  - Epochs: 25
  - Time per epoch: 45 minutes
- **Use Case**: Generation of small molecules (less than 100 elements)

---

### 6. **Model: `insects_smiles_v1.pth`**

- **Dataset**: Insect-specific drug-like molecules from ChEMBL.
- **Description**: This model was trained on SMILES strings representing molecules that act as insecticides or insect-targeting drugs, sourced from the ChEMBL database. It is specialized for generating molecules with potential insecticidal activity.
- **Training Details**:
  - Epochs: 50
  - Time per epoch: 5 minutes
- **Use Case**: Generation of insect-targeting drug-like molecules.

---

## Usage

To use any of the trained models for molecule generation, follow these steps:

1. Navigate to the `models/generator` directory:

   ```bash
   cd models/generator
   ```
2. Generate molecules using a trained model:

```bash
python src/predictions_RNN.py --model_path <models/generator/SELECTED_MODEL> ...
```
