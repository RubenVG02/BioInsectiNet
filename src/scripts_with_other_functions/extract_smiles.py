import pandas as pd

def extract_tsv_smiles(path_file, sep, usecols):
    df = pd.read_csv(path_file, sep=sep, usecols=usecols)
    df = df.dropna()
    return df[usecols[0]].tolist()

def write_smiles(smiles, path_file):
    with open(path_file, 'w') as f:
        for smile in smiles:
            f.write(f"{smile}\n")

def remove_len(smiles):
    return [smile for smile in smiles if len(smile) < 150]

def remove_duplicates(smiles):
    return list(set(smiles))


if __name__ == "__main__":

    smiles = extract_tsv_smiles("data/BindingDB_All.tsv", sep="\t", usecols=["Ligand SMILES"])
    smiles = remove_duplicates(smiles)
    write_smiles(smiles, "data/bindingDB_smiles_All.txt")
    smiles = remove_len(smiles)
    write_smiles(smiles, "data/bindingDB_smiles_filtered.txt")
    print("Done!")