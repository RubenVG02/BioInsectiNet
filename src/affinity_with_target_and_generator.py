
from mega import Mega
import csv

from pretrained_rnn import generator
from check_affinity import calculate_affinity

import pandas as pd

from rdkit import Chem
from rdkit.Chem import Draw
import os
import sys
import base64
import qrcode

#To import sascore to use the Accessibility Score
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

target = "MTSVMSHEFQLATAETWPNPWPMYRALRDHDPVHHVVPPQRPEYDYYVLSRHADVWSAARDH" #Example target sequence, you can change it to your target of interest


def create_file(name_file, headers=["smiles", "IC50", "score"]):
    """
    Creates a .csv file and writes the column headers into it.
    
    Parameters:
        name_file (str): The name of the file to be created.
        headers (list): The column headers to be written to the file.
    """
    with open(f"{name_file}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

def upload_mega(name_file):
    '''
    Upload a file to Mega.nz and obtain a download link.

    Parameters:
        name_file (str): Name of the file to be uploaded. The file should be in the same directory as the script.
    
    '''
    #Use your own mega account
    mail = "your_email@example.com"
    pwd = "your_password"
    mega=Mega()
    mega._login_user(email=mail, password=pwd)
    upload = mega.upload(f"{name_file}.csv")
    link=mega.get_upload_link(upload)
    print("Download link:", link)
    return link
    
def draw_best(lowest_ic50, ic50, smiles, name_file):
    '''
    Draw the best molecule obtained against the selected FASTA target. The image will be saved in the results_examples folder with the name best_molecule_{name_file}.jpg
        
      Parameters:
        lowest_ic50 (float): Lowest IC50 value.
        ic50_list (list): List of IC50 values.
        smiles (list): List of SMILES strings.
        name_file (str): Name of the file where the image will be saved.
    '''
    index = ic50.index(lowest_ic50)
    best_smile = smiles[index]
    molecule = Chem.MolFromSmiles(best_smile)
    if molecule is not None:
        img = Draw.MolToImage(molecule, size=(400, 300))
        img.save(f"results_examples/best_molecule_{name_file}.jpg")

def find_candidates(target=target, name_file_destination="", upload_to_mega=True, draw_minor=True, max_molecules=5, db_smiles=True, path_db_smiles=r"examples/generated_molecules\generated_molecules.txt", accepted_value=1000, generate_qr=True):
    '''
    Generate molecules using an RNN model, compare their affinity with a target, and obtain a synthesis score.

    Parameters:
        target (str): Target sequence for affinity calculation.
        name_file_destination (str): Name of the CSV file to save results.
        upload_to_mega (bool): Upload the CSV to Mega.nz and get a download link. Defaults to True.
        draw_minor (bool): Save an image of the molecule with the best affinity. Defaults to True.
        db_smiles (bool): Use SMILES from a .txt file. Defaults to True.
        path_db_smiles (str): Path to the .txt file with SMILES. Required if db_smiles=True.
        accepted_value (float): Affinity value below which a molecule is considered valid. Defaults to 1000.
        max_molecules (int): Maximum number of valid molecules to include in the output file as hit. Defaults to 5.
        generate_qr (bool): Generate a QR code for the Mega link. Defaults to True.
    

    '''
    ic50 = []
    smiles = []
    score = []
    create_file(name_file=name_file_destination)

    num_generated = 0
    while num_generated < max_molecules:
        if not db_smiles:
            generated = generator(number_generated=10, img_druglike=False)
            smiles.extend(generated)
        else:
            with open(path_db_smiles, "r") as file:
                generated = [line.strip() for line in file]
            smiles.extend(generated)

        for smile in smiles:
            molecule = Chem.MolFromSmiles(smile)
            if molecule:
                sascore = sascorer.calculateScore(molecule)
                score.append(sascore)
                smile = smile.replace("@", "").replace("/", "")
                try:
                    ic50_prediction = calculate_affinity(smile=smile, fasta=target, path_model=r"")
                    ic50_value = float(ic50_prediction)
                    if ic50_value < accepted_value:
                        num_generated += 1
                    ic50.append(ic50_value)
                except Exception as e:
                    print(f"Error calculating affinity for SMILES {smile}: {e}")
                    ic50.append(999999999999999)  # Large value to denote invalid predictions

                if num_generated >= max_molecules:
                    break

    ic50_menor = min(ic50)
    combination = list(zip(smiles, ic50, score))
    
    with open(f"{name_file_destination}.csv", "a", newline="") as file:
        writer = csv.writer(file)
        lines = open(f"{name_file_destination}.csv", "r").read()
        for entry in combination:
            if str(entry[1]) not in lines and entry[1] < accepted_value:
                writer.writerow(entry)
                
    if upload_to_mega:
        link = upload_mega(name_file=name_file_destination)
        if generate_qr:
            qr = qrcode.make(link)
            qr.save(f"qr_img/qr_{name_file_destination}.png")

    if draw_minor:
        draw_best(ic50_menor, ic50, smiles, name_file_destination)


