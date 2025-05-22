import pandas as pd

def extract_smiles_from_csv(input_csv, output_txt, separator=";"):
    """ 
    Extract SMILES from a CSV file and save them to a text file.

    Parameters:
        input_csv (str): Name of the input CSV file (without extension).
        output_txt (str): Name of the output text file (without extension).
        separator (str): Separator used in the CSV file.
    """
    df = pd.read_csv(f"{input_csv}", sep=separator, low_memory=False, on_bad_lines="skip")
    print(df.columns.names)
    unique_smiles = df["Smiles"].unique()
    with open(f"{output_txt}", "w") as file:
        for smile in unique_smiles:
            if smile != "nan" or smile is not None: # Probably caused due to skipping bad lines 
                file.write(f"{smile}\n")

def clean_CNN_data(input_tsv, output_csv, col_smiles="Ligand SMILES", col_ic50="IC50 (nM)", col_seq="BindingDB Target Chain Sequence"):
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
    
    print(f"Loading data from {input_tsv}...")

    if not input_tsv.endswith(".tsv"):
        input_tsv += ".tsv"
    elif not output_csv.endswith(".csv"):
        output_csv += ".csv"

    df = pd.read_csv(f"{input_tsv}", sep="\t", usecols=[col_smiles, col_ic50, col_seq], on_bad_lines="skip", low_memory=False)

    if df.empty:
        print(f"Warning: The file {input_tsv} is empty or contains only bad lines.")
        return None

    print("Cleaning and filtering data...")

    df.rename(columns={col_smiles: "Smiles", col_ic50: "IC50", col_seq: "Sequences"}, inplace=True)

    df.dropna(inplace=True)

    df["IC50"] = pd.to_numeric(df["IC50"].str.replace(r"[<>]", "", regex=True), errors='coerce')

    # We use a max length of 2048 on the FASTA seqs to ensure the embedding has a fixed size that matches the input requirements of the affinity model.
    df.query("0 < IC50 < 1000000 and Smiles.str.len() < 150 and Sequences.str.len() < 2048", inplace=True)

    df["Sequences"] = df["Sequences"].str.upper()

    df.drop_duplicates(subset=["Smiles"], inplace=True)

    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Saving cleaned data to {output_csv}...")
    df.to_csv(f"{output_csv}", index=False, sep=",", float_format="%.6g")

    print("Cleaning process completed!")
    return f"{output_csv}"


clean_CNN_data("data/BindingDB_all.tsv", "data/BindingDB_cleaned_2048")