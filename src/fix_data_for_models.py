import pandas as pd
from chembl_webresource_client.new_client import new_client
import csv

def get_target_smiles(target_id, output_csv="data1"):
    """
    Retrieve all SMILES for a given target and store the data in a CSV file.

    Parameters:
        target_id (str): The CHEMBL ID of the target.
        output_csv (str): Name of the output CSV file (without extension).
    
    Returns:
        pd.DataFrame: DataFrame containing the retrieved data.
    """
    activity_client = new_client.activity
    results = activity_client.filter(target_chembl_id=target_id).filter(standard_type="IC50")
    df = pd.DataFrame.from_dict(results)
    df.to_csv(f"{output_csv}.csv", index=False)
    return df



def clean_and_prepare_data(input_csv="drugs", output_csv="cleaned_data"):
    """
    Clean the data and prepare it for the RNN model, saving the result to a CSV file.

    Parameters:
        input_csv (str): Name of the input CSV file (without extension).
        output_csv (str): Name of the output CSV file (without extension).
    
    Returns:
        str: Name of the output CSV file.
    """
    df = pd.read_csv(f"{input_csv}.csv", sep=";", index_col=False)
    df_clean = df.dropna(subset=["Standard Value", "Smiles"])
    df_clean = df_clean.drop_duplicates(subset=["Smiles"])
    df_clean = df_clean[df_clean["Standard Value"] > 0]
    df_clean = df_clean.reset_index(drop=True)
    
    columns_to_keep = ['Molecule ChEMBL ID', 'Smiles', 'Standard Value']
    df_final = df_clean[columns_to_keep]
    df_final.to_csv(f"{output_csv}.csv", index=False, sep=",")
    
    print(f"Cleaned data: {df_final.shape[0]} records.")
    return f"{output_csv}.csv"



def extract_smiles_from_csv(input_csv="500k_dades", output_txt="smiles_list", separator=","):
    """
    Extract SMILES from a CSV file and save them to a text file.

    Parameters:
        input_csv (str): Name of the input CSV file (without extension).
        output_txt (str): Name of the output text file (without extension).
        separator (str): Separator used in the CSV file.
    """
    df = pd.read_csv(f"{input_csv}.csv", sep=separator, low_memory=False)
    unique_smiles = df["Smiles"].unique()
    with open(f"{output_txt}.txt", "w") as file:
        for smile in unique_smiles:
            file.write(f"{smile}\n")

def clean_affinity_data(input_tsv="inh", output_csv="cleaned_affinity_data", col_smiles="Ligand SMILES", col_ic50="IC50 (nM)", col_seq="BindingDB Target Chain Sequence"):
    """
    Clean affinity data and save it to a CSV file.

    Parameters:
        input_tsv (str): Name of the input TSV file (without extension).
        output_csv (str): Name of the output CSV file (without extension).
        col_smiles (str): Name of the column containing SMILES.
        col_ic50 (str): Name of the column containing IC50 values.
        col_seq (str): Name of the column containing sequences.
    
    Returns:
        str: Name of the output CSV file.
    """
    df = pd.read_csv(f"{input_tsv}.tsv", sep="\t", on_bad_lines="skip", low_memory=False)
    
    df_clean = df[[col_smiles, col_ic50, col_seq]].dropna()
    df_clean.columns = ["Smiles", "IC50", "sequence"]
    
    df_clean["IC50"] = df_clean["IC50"].str.strip().str.replace(r"[<>]", "", regex=True).astype(float)
    df_clean = df_clean[(df_clean["IC50"] > 0) & (df_clean["IC50"] < 1000000)]
    df_clean = df_clean[df_clean["Smiles"].str.len() < 100]
    df_clean = df_clean[df_clean["sequence"].str.len() < 5000]
    df_clean["sequence"] = df_clean["sequence"].str.upper()
    
    df_clean = df_clean.drop_duplicates(subset=["Smiles"])
    df_clean = df_clean.sample(frac=1).reset_index(drop=True)
    
    df_clean.to_csv(f"{output_csv}.csv", index=False, sep=",")
    return f"{output_csv}.csv"
