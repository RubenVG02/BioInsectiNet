import pandas as pd
import numpy as np

def extract_tsv_smiles(path_file, sep, usecols):
    df = pd.read_csv(path_file, sep=sep, usecols=usecols)
    df = df.dropna()
    return df[usecols[0]].tolist()

def write_smiles(smiles, path_file):
    with open(path_file, 'w') as f:
        for smile in smiles:
            f.write(f"{smile}\n")

def remove_len(smiles):
    return [smile for smile in smiles if len(smile) > 130]

def remove_duplicates(smiles):
    return list(set(smiles))

def get_percentile_90(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    lengths = [len(line.strip()) for line in lines]
    
    percentile_90 = np.percentile(lengths, 90)
    print(f"90th percentile: {percentile_90}")
    
    return percentile_90

with open("data\smiles\combined_longest.txt", 'r') as file:
    smiles = file.readlines()

smiles = [smile.strip() for smile in smiles]
smiles = remove_duplicates(smiles)

with open("data\smiles\combined_longest_no_duplicates.txt", 'w') as file:
    for smile in smiles:
        file.write(f"{smile}\n")