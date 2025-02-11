import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Draw
from tqdm import tqdm
import os
from rdkit import RDLogger
import csv
import argparse

import sys
from rdkit.Chem import RDConfig
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer

from train_RNN_generation import load_smiles, pack_padded_sequence, pad_packed_sequence

RDLogger.DisableLog('rdApp.*')  # To deactivate RDKit warnings during molecule generation


HIDDEN_SIZE = 256
EMBEDDING_DIM = 256
NUM_LAYERS = 3
DROPOUT = 0.2
PAD_TOKEN = "<PAD>"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate drug-like molecules using a pretrained RNN model.")
    parser.add_argument("--model_path", type=str, default="models\generator\chembl_smiles_longest_v1.pth", help="Path to the trained model. By default, it uses the pretrained model trained on the ChEMBL dataset.")
    parser.add_argument("--data_path", type=str, help="Path to the SMILES dataset. By default, it uses the dataset used to train the model.")
    parser.add_argument("--save_dir", type=str, default="generated_molecules", help="Directory to save the generated molecules. By default, it saves the molecules in the 'generated_molecules' directory.")
    parser.add_argument("--num_molecules", type=int, default=250, help="Number of molecules to generate. By default, it generates 250 molecules.")
    parser.add_argument("--min_length", type=int, default=50, help="Minimum length of generated SMILES. By default, it generates molecules with a minimum length of 20 characters.")
    parser.add_argument("--max_length", type=int, default=500, help="Maximum length of generated SMILES. By default, it generates molecules with a maximum length of 500 characters.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. By default, it uses a temperature of 1.0.")
    parser.add_argument("--save_images", action='store_true', help="Save generated molecule images. By default, it does not save the images.")
    return parser.parse_args()

class ImprovedSMILESGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, char_to_idx):
        super(ImprovedSMILESGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.char_to_idx = char_to_idx
        self.idx_to_char = {idx: char for char, idx in char_to_idx.items()}

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=char_to_idx[PAD_TOKEN])
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, lengths, hidden=None):
        x = self.embedding(x)
        x = self.dropout_layer(x)
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, hidden = self.lstm(x, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=max(lengths))
        output = self.dropout_layer(output)
        output = self.fc(output)
        return output, hidden

def load_model(model_path, char_to_idx, vocab_size):
    model = ImprovedSMILESGenerator(vocab_size, EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, char_to_idx)
    model.load_state_dict(torch.load(model_path))
    model.eval()  
    return model

def generate_smiles(model, start_char, max_length, temperature=1.0):
    device = next(model.parameters()).device
    model.eval()

    current_char = start_char
    hidden = None
    generated_seq = [current_char]

    for _ in range(max_length):
        input_tensor = torch.tensor([[model.char_to_idx[current_char]]], dtype=torch.long).to(device)
        lengths = torch.tensor([1], dtype=torch.long).to(device)

        # Pasar por el modelo
        with torch.no_grad():
            output, hidden = model(input_tensor, lengths, hidden)
            output = output[:, -1, :]  



        output = output / temperature # To control the randomness of the generated sequences
        probs = torch.softmax(output, dim=-1)
        next_char_idx = torch.multinomial(probs, num_samples=1).item()

        next_char = model.idx_to_char[next_char_idx]

        # End of sequence if newline character, as specified during the training step
        if next_char == '\n':
            break

        generated_seq.append(next_char)
        current_char = next_char
    return "".join(generated_seq)

def is_druglike(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False  

    # Lipinski's Rule of Five (https://en.wikipedia.org/wiki/Lipinski%27s_Rule_of_Five)
    mol_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    h_bond_donors = Lipinski.NumHDonors(mol)
    h_bond_acceptors = Lipinski.NumHAcceptors(mol)

    return (mol_weight <= 500 and
            logp <= 5 and
            h_bond_donors <= 5 and
            h_bond_acceptors <= 10)

def save_molecule_image(smiles, filename):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        Draw.MolToFile(mol, filename)

def generate_druglike_molecules(model_path, char_to_idx, vocab_size, num_molecules=100, min_length = 30, max_length=100, temperature=1.0, save_images=False, output_dir="generated_molecules"):
    model = load_model(model_path, char_to_idx, vocab_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    os.makedirs(output_dir, exist_ok=True)

    if save_images:
        image_dir = os.path.join(output_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

    smiles_file = os.path.join(output_dir, "generated_molecules.smi")
    csv_file = os.path.join(output_dir, "generated_molecules.csv")
    previous_molecules = set()

    
    if os.path.exists(csv_file):
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  
            previous_molecules.update(row[0] for row in reader)  

    if os.path.exists(smiles_file):
        with open(smiles_file, "r") as f:
            previous_molecules.update(line.strip() for line in f.readlines())

    druglike_molecules = list(previous_molecules)  
    start_char = list(char_to_idx.keys())[0]  

    with tqdm(total=num_molecules, desc="Generating molecules") as pbar:
        while len(druglike_molecules) - len(previous_molecules) < num_molecules:

            

            smiles = generate_smiles(model, start_char, max_length, temperature)
            smiles = smiles.replace('\n', '')  

            if smiles in previous_molecules or smiles in druglike_molecules:
                continue  # Ignoring duplicates and molecules that are not drug-like

            
            is_druglike_flag = is_druglike(smiles)
            if not is_druglike_flag or len(smiles) < min_length:
                continue 

            sas_score = calculate_sas(smiles)  

            # Every time a molecule is generated and is drug-like, it is saved, and the progress bar is updated, avoiding duplicates and molecules that are not drug-like
            druglike_molecules.append(smiles)
            pbar.update(1)

            if save_images:
                image_filename = os.path.join(image_dir, f"molecule_{len(druglike_molecules)}.png")
                save_molecule_image(smiles, image_filename)


    with open(smiles_file, "w") as f:
        for smiles in druglike_molecules:
            f.write(smiles + "\n")


    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["SMILES", "Drug-like", "SAS Score"])  
        for smiles in druglike_molecules:
            sas_score = calculate_sas(smiles)
            writer.writerow([smiles, True, sas_score])  

    print(f"Drug-like molecules saved to '{smiles_file}'.")
    print(f"CSV file saved to '{csv_file}'.")
    if save_images:
        print(f"Molecule images saved to '{image_dir}'.")

    return druglike_molecules


def calculate_sas(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return sascorer.calculateScore(mol)
    return None


if __name__ == "__main__":
    args = parse_arguments()
    if args.data_path is None: 
        args.data_path = "data/smiles" + os.sep + os.path.basename(args.model_path).split(".")[0].rsplit("_", 1)[0] + ".txt"
    print(args.data_path)
    smiles_list = load_smiles(args.data_path)

    # To include the PAD token and newline character
    unique_chars = sorted(set("".join(smiles_list)) | {'\n', PAD_TOKEN})

    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    vocab_size = len(char_to_idx)  

    

    druglike_molecules = generate_druglike_molecules(
        model_path=args.model_path,
        char_to_idx=char_to_idx,
        vocab_size=vocab_size,
        num_molecules=args.num_molecules,
        min_length=args.min_length,
        max_length=args.max_length,
        temperature=args.temperature,
        save_images=args.save_images,
        output_dir=args.save_dir
    )
