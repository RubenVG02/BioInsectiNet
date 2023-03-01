import pandas as pd
from chembl_webresource_client.new_client import new_client
import csv


def selection_mol(id, name_file="data1", name_df="df"):
    '''
    Function to obtain all the smiles of a given target, and store them into a csv file
    
    Parameters:
        -id: CHEMBL ID
        -name_file: Name of the file (by default, "data1")
        -name_df: Name df (by default, "df")
    '''
    act = new_client.activity
    # IC50: amount of inhibitor subs you need to inhibit at 50%
    res = act.filter(target_chembl_id=id).filter(standard_type="IC50")
    name_df = pd.DataFrame.from_dict(res)
    name_df.to_csv(f"{name_file}.csv", index_label=False)
    return name_df


def clean_data_rnn(name_file="drugs", name_file2="dades_netes"):
    ''' 
    Function to clean the data and store it into a csv file, which will be ready to be used for the RNN model
    
    Parameters:
        -name_file: input file name 
        -name_file2: name of the net archive
    '''
    arx = pd.read_csv(f"{name_file}.csv", sep=";", index_col=False)
    datfr = arx[arx["Standard Value"].notna()]
    datfr = datfr[arx.Smiles.notna()]
    df_no_dup = datfr.drop_duplicates(['Smiles'])
    selecc = ['Molecule ChEMBL ID', 'Smiles', 'Standard Value']
    df2 = df_no_dup[selecc]
    print(df2.value_counts())

    # Standard Value smaller than 0 are deleted
    df2 = df2[df2["Standard Value"] > 0]
    df2 = df2.reset_index(drop=True)
    # This is the 
    df2.to_csv(f"{name_file2}.csv", index=False, sep=",")

    # Quants elements utilitzes per entrenar el model
    print(df2.value_counts())

    return f"{name_file2}.csv"


def obtain_smiles(origin_file="500k_dades", destination_txt="smiles_22", separator=","):
    '''
        Parameters:
        -origin_file: file from which we will obtain the smiles
        -destination_file: file where the smiles will be saved
        -separator: separator of the file (by default, ",")
    '''
    dades = pd.read_csv(f"{origin_file}.csv", sep=separator,low_memory=False)
    llista_smiles = dades["Smiles"].unique()
    with open(f"{destination_txt}.txt", "w") as f:
        for line in llista_smiles:
            f.write(str(line) + "\n")


def neteja_dades_afinitat(name_file="inh", destination_file="cnn_file_fixed", col_smiles="Ligand SMILES", col_ic50="IC50 (nM)", col_seq="BindingDB Target Chain Sequence"):
    '''
        Parameters:
         -name_file: Name of the source file to modify, in tsv format
         -target_name: Name of the target file to be created
         -col_smiles: Columns from which you want to get the smiles
         -col_ic50: Columns from which to obtain the ic50
         -col_seq: Columns from which to obtain the sequences

    '''
    with open(f"{name_file}.tsv", "r", encoding="utf8") as file:
        df = pd.read_csv(file, sep="\t", on_bad_lines="skip",
                         low_memory=False, nrows=1000000)
        columna_cadena = df[col_seq]
        columna_50 = df[col_ic50]
        columna_smiles = df[col_smiles]

    cadena = []
    for i in columna_cadena:
        cadena.append(i)
    ic50 = []
    for i in columna_50:
        ic50.append(i)
    smiles = []
    for i in columna_smiles:
        smiles.append(i)

    headers = ["Smiles", "IC50", "sequence"]
    listas = [smiles, ic50, cadena]
    with open(f"{destination_file}.csv", "w", encoding="utf8") as archivo:
        write = csv.writer(archivo)
        write.writerow(headers)
        write.writerows(zip(*listas))

    arx = pd.read_csv(f"{destination_file}.csv", sep=",")
    arx["IC50"] = arx["IC50"].str.strip()
    arx = arx.dropna(how="any").reset_index(drop=True)
    # print(datfr)
    df_no_dup = arx.drop_duplicates(['smiles'])
    df_no_dup["IC50"] = df_no_dup["IC50"].str.replace(r"[<>]", "", regex=True)
    df_no_dup["IC50"] = df_no_dup["IC50"].astype(float)
    df_no_dup = df_no_dup[df_no_dup["IC50"] < 1000000]
    df_no_dup = df_no_dup[df_no_dup["Smiles"].str.len() < 100]
    df_no_dup = df_no_dup[df_no_dup["sequence"].str.len() < 5000]
    #smiles_without_symbol = []
    ''' for i in df_no_dup["smiles"]:
        smiles_sense_simbols.append(i.replace("@", "").replace("\\",
                                                               "").replace("/", "").replace(".", ""))'''

    '''df_no_dup["smiles"] = smiles_sense_simbols
    del smiles_sense_simbols'''
    df_no_dup["sequence"] = df_no_dup["sequence"].apply(
        lambda x: x.upper())
    df_no_dup = df_no_dup.sample(frac=1).reset_index(drop=True)
    df_no_dup.to_csv(f"{destination_file}.csv", index=False, sep=",")
    return f"{destination_file}.csv"


obtain_smiles("aaa", "insectos", separator=";")