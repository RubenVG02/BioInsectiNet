import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_absolute_error, r2_score
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
import argparse
import subprocess

# Disable RDKit warnings during preprocessing
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models/checkpoints", exist_ok=True)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, default="data/cleaned_data.csv", help="Path to the raw data file")
    parser.add_argument("--preprocessed_data_path", type=str, default="data/preprocessed_data.pkl", help="Path to the preprocessed data file. If it doesn't exist, the data will be preprocessed and saved in this path")
    parser.add_argument("--study_name", type=str, default="cnn_affinity", help="Name of the study. It will be used to save the study results")
    parser.add_argument("--storage", type=str, default="sqlite:///models/cnn_affinity.db", help="Storage for the study results (SQLite database)")
    parser.add_argument("--visualize", action="store_true", help="Visualize the study results using optuna built-in dashboard. Default is False")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials for the optimization. Default is 20")
    return parser.parse_args()

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits))
    return np.zeros(n_bits)

def fasta_to_bert_embedding(fasta, tokenizer, model, max_length=1024):
    inputs = tokenizer(fasta, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs.to(device))
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()  

def smiles_to_chemberta_embedding(smiles, tokenizer, model, max_length=512):
    inputs = tokenizer(smiles, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs.to(device))
    return outputs.last_hidden_state[:, 0, :].cpu().numpy()  
def preprocess_data(csv_path, tokenizer_bert, model_bert, tokenizer_chem, model_chem, save_path=None):
    df = pd.read_csv(csv_path)
    
    df = df[df['IC50'] <= 10000]
    df = df[(df['IC50'] >= df['IC50'].quantile(0.05)) & (df['IC50'] <= df['IC50'].quantile(0.95))]
    
    tqdm.pandas()
    
    df['Smiles'] = df['Smiles'].progress_apply(lambda x: x if Chem.MolFromSmiles(x) else None)
    
    df['Sequences'] = df['Sequences'].progress_apply(lambda x: x if set(x).issubset(set("ACDEFGHIKLMNPQRSTVWY")) else None)
    
    df = df.dropna(subset=['Smiles', 'Sequences'])
    
    df['IC50'] = np.log(df['IC50'])
    
    df['smiles_embedding'] = df['Smiles'].progress_apply(lambda x: smiles_to_chemberta_embedding(x, tokenizer_chem, model_chem))
    
    df['bert_embedding'] = df['Sequences'].progress_apply(lambda x: fasta_to_bert_embedding(x, tokenizer_bert, model_bert))
    
    df['smiles_embedding'] = df['smiles_embedding'].apply(lambda x: (x - np.mean(x)) / np.std(x))

    df['bert_embedding'] = df['bert_embedding'].apply(lambda x: (x - np.mean(x)) / np.std(x))
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_pickle(save_path)
        print(f"Preprocessed data saved in {save_path}")
    
    return df

def load_preprocessed_data(pkl_path):
    print("Loading preprocessed data...")
    if os.path.exists(pkl_path):
        return pd.read_pickle(pkl_path)


class IC50Dataset(Dataset):
    def __init__(self, df):
        self.smiles_embeddings = np.vstack(df['smiles_embedding'].values)
        self.bert_embeddings = np.vstack(df['bert_embedding'].values)
        self.targets = df['IC50'].values.astype(np.float32)
    
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return (
            torch.tensor(self.smiles_embeddings[idx], dtype=torch.float32),
            torch.tensor(self.bert_embeddings[idx], dtype=torch.float32),
            torch.tensor(self.targets[idx], dtype=torch.float32)
        )


class CombinedModel(nn.Module):
    def __init__(self, chem_dim=768, bert_dim=1024, hidden_dim=512, dropout=0.3, num_layers=2):
        super(CombinedModel, self).__init__()
        
        self.fc_chem = nn.Sequential(
            nn.Linear(chem_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers - 1)]
        )
        
        self.fc_bert = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers - 1)]
        )
        
        self.fc_combined = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, chem_emb, bert_emb):
        chem_out = self.fc_chem(chem_emb)
        bert_out = self.fc_bert(bert_emb)
        combined = torch.cat([chem_out, bert_out], dim=1)
        combined_out = self.fc_combined(combined)
        return self.output(combined_out)
    

def objective(trial, train_dataset, val_dataset):
    hidden_dim = trial.suggest_int("hidden_dim", 128, 1024)
    dropout = trial.suggest_float("dropout", 0.05, 0.3)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    num_layers = trial.suggest_int("num_layers", 1, 4)  
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64]) 

    model = CombinedModel(hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layers).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_model_with_optuna(trial, model, train_loader, val_loader)

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.epochs_no_improve = 0

def train_model_with_optuna(trial, model, train_loader, val_loader, epochs=50, scheduler=None):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "val_r2": [], "val_pearson": [], "val_spearman": []}
    
    early_stopping = EarlyStopping(patience=5, delta=0.01)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for chem_emb, bert_emb, target in train_loader:
            chem_emb, bert_emb, target = chem_emb.to(device), bert_emb.to(device), target.to(device)
            
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                output = model(chem_emb, bert_emb).squeeze()
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        history["train_loss"].append(avg_train_loss)
        
        model.eval()
        val_loss, all_preds, all_targets = 0.0, [], []
        with torch.no_grad():
            for chem_emb, bert_emb, target in val_loader:
                chem_emb, bert_emb, target = chem_emb.to(device), bert_emb.to(device), target.to(device)
                with autocast(device_type=device.type):
                    output = model(chem_emb, bert_emb).squeeze()
                loss = criterion(output, target)
                val_loss += loss.item()
                all_preds.extend(output.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        all_preds, all_targets = np.array(all_preds), np.array(all_targets)
        val_mae = mean_absolute_error(all_targets, all_preds)
        val_r2 = r2_score(all_targets, all_preds)
        val_pearson = np.corrcoef(all_targets, all_preds)[0, 1]
        val_spearman = spearmanr(all_targets, all_preds).correlation
        avg_val_loss = val_loss / len(val_loader)
        
        history["val_loss"].append(avg_val_loss)
        history["val_mae"].append(val_mae)
        history["val_r2"].append(val_r2)
        history["val_pearson"].append(val_pearson)
        history["val_spearman"].append(val_spearman)
        
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "models/checkpoints/best_model.pth")
        
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    return best_val_loss

if __name__ == "__main__":
    args = args_parser()
    # We use prot_bert and ChemBERTa-zinc-base-v1 as the pretrained models.
    tokenizer_bert = AutoTokenizer.from_pretrained("Rostlab/prot_bert")
    model_bert = AutoModel.from_pretrained("Rostlab/prot_bert").to(device)
    tokenizer_chem = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
    model_chem = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1").to(device)
    
    
    df = load_preprocessed_data(args.preprocessed_data_path)
    if df is None:
        df = preprocess_data(args.raw_data_path, tokenizer_bert, model_bert, tokenizer_chem, model_chem, save_path=args.preprocessed_data_path)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = IC50Dataset(train_df)
    val_dataset = IC50Dataset(val_df)
    


    try:
        study = optuna.load_study(study_name=args.study_name, storage=args.storage)
        print("Study loaded.")
    except:
        print(f"No study found. Creating a new study: {args.study_name}")
        study = optuna.create_study(direction="minimize", study_name=args.study_name, storage=args.storage)

    
    if args.visualize:
        if not "sqlite:///" in args.storage:
            args.storage = f"sqlite:///{args.storage}"
        print("Launching Optuna dashboard...")
        #stdout=subprocess.DEVNULL to hide the website output logs, which are constantly updating during the optimization.
        subprocess.Popen(f"optuna-dashboard {args.storage}", shell=True, stdout=subprocess.DEVNULL) 
        print("Optuna dashboard successfully launched, to access it, go to 'http://localhost:8080' in your browser")
    
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset), n_trials=args.n_trials)
    
    print(f"Best hyperparameters found: {study.best_params}")
    print(f"Best value found: {study.best_value}")

    
