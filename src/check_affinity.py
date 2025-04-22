import torch
from train_cnn_affinity import CustomModel 
from tqdm import tqdm
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

elements_smiles = np.load("models/elements_smiles.npy")
elements_fasta = np.load("models/elements_fasta.npy")

int_smiles = {char: idx + 1 for idx, char in enumerate(elements_smiles)}
int_fasta = {char: idx + 1 for idx, char in enumerate(elements_fasta)}

def convert_sequences(smiles_list, fasta_list, max_smiles=150, max_fasta=5000):
    smiles_w_numbers = []
    fasta_w_numbers = []

    print("Processing SMILES...")
    for smile in tqdm(smiles_list, desc="SMILES", unit="seq"):
        smile_numbers = [int_smiles.get(char, 0) for char in smile][:max_smiles]
        smile_numbers.extend([0] * (max_smiles - len(smile_numbers)))
        smiles_w_numbers.append(smile_numbers)

    print("Processing FASTA...")
    for fasta in tqdm(fasta_list, desc="FASTA", unit="seq"):
        fasta_numbers = [int_fasta.get(char, 0) for char in fasta][:max_fasta]
        fasta_numbers.extend([0] * (max_fasta - len(fasta_numbers))) 
        fasta_w_numbers.append(fasta_numbers)

    return np.array(smiles_w_numbers), np.array(fasta_w_numbers)


def predict_affinity(smile, fasta, path_model):
    model = CustomModel(150, 5000, len(elements_smiles), len(elements_fasta))
    model.load_state_dict(torch.load(path_model))
    model.eval()
    model.to(device)

  
    smile_tensor, fasta_tensor = convert_sequences([smile], [fasta]) 
    smile_tensor = torch.tensor(smile_tensor).to(device)
    fasta_tensor = torch.tensor(fasta_tensor).to(device)

    with torch.no_grad():
        affinity = model(smile_tensor, fasta_tensor).item()

    return affinity


smile = "Nc1ccccc1NC(=O)c1ccc(C=C2CN(Cc3cccnc3)C2)c(Cl)c1"
fasta = "MAQTQGTRRKVCYYYDGDVGNYYYGQGHPMKPHRIRMTHNLLLNYGLYRKMEIYRPHKANAEEMTKYHSDDYIKFLRSIRPDNMSEYSKQMQRFNVGEDCPVFDGLFEFCQLSTGGSVASAVKLNKQQTDIAVNWAGGLHHAKKSEASGFCYVNDIVLAILELLKYHQRVLYIDIDIHHGDGVEEAFYTTDRVMTVSFHKYGEYFPGTGDLRDIGAGKGKYYAVNYPLRDGIDDESYEAIFKPVMSKVMEMFQPSAVVLQCGSDSLSGDRLGCFNLTIKGHAKCVEFVKSFNLPMLMLGGGGYTIRNVARCWTYETAVALDTEIPNELPYNDYFEYFGPDFKLHISPSNMTNQNTNEYLEKIKQRLFENLRMLPHAPGVQMQAIPEDAIPEESGDEDEDDPDKRISICSSDKRIACEEEFSDSEEEGEGGRKNSSNFKKAKRVKTEDEKEKDPEEKKEVTEEEKTKEEKPEAKGVKEEVKLA"
path_model = r"models\checkpoints\best_model_20250206-132136.pth"

affinity = predict_affinity(smile, fasta, path_model)
affinity = np.power(10, affinity) # As we performed a log transformation in the training, we need to revert it
affinity = round(affinity, 2)
print(f"Predicted affinity: {affinity}")
