
from mega import Mega
import csv

from predictions_RNN import generate_druglike_molecules, load_unique_chars_dict
from check_affinity import predict_affinity, get_best_trial

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
import os
import sys
import qrcode
import argparse
from tqdm import tqdm
import re

#To import sascore to use the Accessibility Score
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def arg_parser():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Find candidates for a given target protein.")
    parser.add_argument("--target", type=str, required=True, help="Target protein sequence in FASTA format.")
    parser.add_argument("--name_file_destination", type=str, required=True, help="Output CSV filename without extension.")
    parser.add_argument("--output_dir", type=str, default="results_examples", help="Directory to save results.")
    parser.add_argument("--upload_to_mega", action='store_true', help="Upload result CSV to Mega.")
    parser.add_argument("--draw_lowest", action='store_true', help="Save image of molecule with best IC50.")
    parser.add_argument("--max_molecules", type=int, default=5, help="Max number of valid molecules to keep.")
    parser.add_argument("--total_generated", type=int, default=1000, help="Total number of molecules to generate.")
    parser.add_argument("--db_smiles", action='store_true', help="Use pre-generated SMILES from file.")
    parser.add_argument("--path_db_smiles", type=str, default=r"generated_molecules\generated_molecules.smi", help="Path to SMILES file.")
    parser.add_argument("--accepted_value", type=float, default=1000, help="IC50 threshold. Molecules below this are considered valid.")
    parser.add_argument("--generate_qr", action='store_true', help="Generate QR code to Mega link.")
    parser.add_argument("--affinity_model_path", type=str, default=r"models\checkpoints\cnn_affinity\trial_1_loss_0.1974.pth", help="Path to the IC50 prediction model.")
    parser.add_argument("--db_affinity_path", type=str, default="models/cnn_affinity.db", help="Path to the database with hyperparameters for the affinity model.")
    parser.add_argument("--study_name", type=str, default="cnn_affinity", help="Name of the study for hyperparameter optimization.")
    parser.add_argument("--generator_model_path", type=str, default=r"models\generator\bindingDB_smiles_filtered_v1.pth", help="Path to the RNN generator model.")
    parser.add_argument("--smiles_to_draw", type=int, default=1, help="Number of top SMILES to draw. Only used if draw_lowest is True. By default, it is set to 1.")

    return parser.parse_args()

def upload_mega(name_file: str):
    """
    Upload a file to Mega.nz and obtain a download link.

    Parameters:
        name_file (str): Name of the file to be uploaded. The file should be in the same directory as the script.
    
    """
    mega=Mega()

    # Make sure to set the environment variables MEGA_EMAIL and MEGA_PASSWORD
    mail = os.environ.get("MEGA_EMAIL")
    pwd = os.environ.get("MEGA_PASSWORD")
    if mail is None or pwd is None:
        raise ValueError("Please set the MEGA_EMAIL and MEGA_PASSWORD environment variables.")
    
    mega._login_user(email=mail, password=pwd)
    upload = mega.upload(f"{name_file}.csv")
    link=mega.get_upload_link(upload)
    print("Download link:", link)
    return link
    
def draw_best(smiles_to_draw, ic50: list, smiles: list, name_file: str, output_dir: str): 
    """
    Draw the top N molecules with the lowest IC50 values for the selected FASTA target.
    The images will be saved in the output_dir with the name:
    best_molecule_{rank}_{name_file}.jpg

    Parameters:
        smiles_to_draw (int): Number of top SMILES to draw.
        ic50 (list): List of IC50 values.
        smiles (list): List of SMILES strings.
        name_file (str): Base name for the output files.
        output_dir (str): Directory where the images will be saved.

    Returns:
        None
    """
    ordered_df = pd.DataFrame({"IC50": ic50, "SMILES": smiles})
    ordered_df = ordered_df.sort_values(by="IC50").reset_index(drop=True)

    smiles_to_draw = min(smiles_to_draw, len(ordered_df))
    for i in range(smiles_to_draw):
        mol = Chem.MolFromSmiles(ordered_df["SMILES"][i])
        if mol is None:
            print(f"[WARNING] Invalid SMILES: {ordered_df['SMILES'][i]}")
            continue
        img = Draw.MolToImage(mol, size=(300, 300))
        img.save(os.path.join(output_dir, f"best_molecule_{i+1}_{name_file}.jpg"))
        print(f"[INFO] Saved image for molecule {i+1} with IC50 {ordered_df['IC50'][i]} to {output_dir}/best_molecule_{i+1}_{name_file}.jpg")



    

def find_candidates(
    target: str,
    name_file_destination: str,
    output_dir: str,
    upload_to_mega: bool,
    draw_lowest: bool,
    max_molecules: int,
    db_smiles: bool,
    path_db_smiles: str,
    accepted_value: float,
    generate_qr,
    affinity_model_path: str,
    generator_model_path: str,
    db_affinity_path: str,
    study_name: str,
    total_generated: int = 1000, # As when we use db_smiles we don't generate molecules, we set the default to 1000
    smiles_to_draw: int = 1
    ): 
    """
    Generate molecules using an RNN model, predict their affinity (IC50) with a target, and store valid hits.

    Parameters:
        target (str): Target protein sequence in FASTA format.
        name_file_destination (str): Output CSV filename without extension.
        upload_to_mega (bool): Upload result CSV to Mega. Defaults to True.
        draw_lowest (bool): Save image of molecule with best IC50. Defaults to True.
        max_molecules (int): Max number of valid molecules to keep. Defaults to 5.
        db_smiles (bool): Use pre-generated SMILES from file. Defaults to True.
        path_db_smiles (str): Path to SMILES file. Required if db_smiles is True.
        accepted_value (float): IC50 threshold. Molecules below this are considered valid. Defaults to 1000.
        generate_qr (bool): Generate QR code to Mega link. Defaults to True.
        affinity_model_path (str): Path to the IC50 prediction model.
        generator_model_path (str): Path to the RNN generator model.
        db_affinity_path (str): Path to the database with hyperparameters for the affinity model.
        study_name (str): Name of the study for hyperparameter optimization.
        total_generated (int): Total number of molecules to generate. Defaults to 1000.
        smiles_to_draw (int): Number of top SMILES to draw. Only used if draw_lowest is True. Defaults to 1.

    Returns:
        None. Saves a CSV file with results, and optionally uploads it to Mega and creates a QR code.
    """

    if db_smiles:
        if not os.path.exists(path_db_smiles):
            raise FileNotFoundError(f"The file {path_db_smiles} does not exist.")
        with open(path_db_smiles, "r") as f:
            generated = [line.strip() for line in f]
            print(f"[INFO] Loaded {len(generated)} SMILES from {path_db_smiles}.")

    else:
        base_name = os.path.basename(generator_model_path)
        data_path = re.sub(r'_v\d+\.pth$', '', base_name) + ".txt"
        print(f"[INFO] Using data path: {data_path} for unique characters dictionary.")

        unique_chars_dict = load_unique_chars_dict("models/unique_chars_dict.json")

        key_found = next((key for key in unique_chars_dict if key.endswith(data_path)), None)
        if key_found:
            unique_chars = unique_chars_dict[key_found]
        else:
            raise ValueError(f"[ERROR] Unique characters for {data_path} not found in the dictionary.")
        print(f"[INFO] Using {len(unique_chars)} unique characters for SMILES generation.")
        char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}


        generated = generate_druglike_molecules(generator_model_path, num_molecules=total_generated, save_images=False, 
                                                 char_to_idx=char_to_idx, max_length=155, vocab_size=len(unique_chars))
        print(f"[INFO] Generated {len(generated)} SMILES using RNN generator.")

    # If the output dir with the name_file_destination does exist, we add a number to the name_file_destination until it is unique
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(os.path.join(output_dir, name_file_destination)):
        i = 1
        while os.path.exists(os.path.join(output_dir, f"{name_file_destination}_{i}")):
            i += 1
        name_file_destination = f"{name_file_destination}_{i}"
        print(f"[INFO] Output directory already exists. Using {name_file_destination} instead.")
    os.makedirs(os.path.join(output_dir, name_file_destination), exist_ok=True)
    csv_path = os.path.join(output_dir, name_file_destination, f"{name_file_destination}.csv")
    file_destination = os.path.join(output_dir, name_file_destination)


    df = pd.DataFrame(columns=["SMILES", "IC50", "SA_Score"])
    df.to_csv(csv_path, index=False) 
    print(f"[INFO] Created new CSV with headers at {csv_path}.")

    results = []

    # We divide the generated molecules into batches of 150 to avoid memory issues
    total_hits = 0
    print(f"[INFO] Starting to process {len(generated)} generated molecules.")
    while total_hits < max_molecules and generated:
        batch = generated[:150]
        generated = generated[150:]

        for smile in tqdm(batch, desc="Processing SMILES"):
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                continue

            sa_score = sascorer.calculateScore(mol)
            cleaned_smile = smile.replace("@", "").replace("/", "")

            try:
                best_trial = get_best_trial(db_affinity_path, study_name=study_name)
                if best_trial is None:
                    raise ValueError(f"[ERROR] No best trial found in {db_affinity_path} for study {study_name}.")
                ic50_pred = float(predict_affinity(smile=cleaned_smile, fasta=target, path_model=affinity_model_path, best_trial=best_trial))
                ic50_pred = 10 ** -ic50_pred
            except Exception as e:
                print(f"Error calculating affinity for {cleaned_smile}: {e}")
                continue

            if ic50_pred < accepted_value:
                if cleaned_smile not in df["SMILES"].values:
                    results.append((cleaned_smile, ic50_pred, sa_score))
                    total_hits += 1
                    if total_hits >= max_molecules:
                        break
    print(f"[INFO] Found {len(results)} valid molecules with IC50 < {accepted_value}.")

    new_df = pd.DataFrame(results, columns=["SMILES", "IC50", "SA_Score"])
    final_df = pd.concat([df, new_df], ignore_index=True)
    final_df.to_csv(csv_path, index=False)

    print(f"[INFO] Results saved to {csv_path}.")
    if draw_lowest and not new_df.empty:
        img_dir = os.path.join(output_dir, name_file_destination)
        draw_best(smiles_to_draw, new_df["IC50"].tolist(), new_df["SMILES"].tolist(), name_file_destination, img_dir)

    if upload_to_mega:
        link = upload_mega(name_file=file_destination)
        if generate_qr:
            qr = qrcode.make(link)
            qr_dir = "qr_img"
            os.makedirs(qr_dir, exist_ok=True)
            qr.save(os.path.join(qr_dir, f"qr_{name_file_destination}.png"))

if __name__ == "__main__":
    # Example fasta sequence
    target = "MTSVMSHEFQLATAETWPNPWPMYRALRDHDPVHHVVPPQRPEYDYYVLSRHADVWSAARDH" 
    args = arg_parser()

    find_candidates(
        target=args.target,
        name_file_destination=args.name_file_destination,
        output_dir=args.output_dir,
        upload_to_mega=args.upload_to_mega,
        draw_lowest=args.draw_lowest,
        max_molecules=args.max_molecules,
        total_generated=args.total_generated,
        db_smiles=args.db_smiles,
        path_db_smiles=args.path_db_smiles,
        accepted_value=args.accepted_value,
        generate_qr=args.generate_qr,
        affinity_model_path=args.affinity_model_path,
        generator_model_path=args.generator_model_path,
        db_affinity_path=args.db_affinity_path,
        study_name=args.study_name,     
        smiles_to_draw=args.smiles_to_draw   
    )
