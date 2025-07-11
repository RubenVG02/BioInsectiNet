import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from rdkit import Chem
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import warnings
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import optuna
import argparse
import subprocess
import json
from sklearn.metrics import mean_squared_error, r2_score

torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()

# Disable RDKit warnings during preprocessing
warnings.filterwarnings("ignore")
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("models/checkpoints", exist_ok=True)


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, default="data/BindingDB_cleaned_1024.csv", help="Path to the raw data file")
    parser.add_argument("--preprocessed_data_path", type=str, default="data/preprocessed_data.pkl", help="Path to the preprocessed data file. If it doesn't exist, the data will be preprocessed and saved in this path")
    parser.add_argument("--study_name", type=str, default="cnn_affinity", help="Name of the study. It will be used to save the study results")
    parser.add_argument("--storage", type=str, default="sqlite:///models/cnn_affinity.db", help="Storage for the study results (SQLite database)")
    parser.add_argument("--visualize", action="store_true", help="Visualize the study results using optuna built-in dashboard. Default is False")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of trials for the optimization. Default is 20")
    parser.add_argument("--create_subfolders", action="store_true", help="Create subfolders for the models and checkpoints. Default is False")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output during training. Default is False")
    return parser.parse_args()




def fasta_to_bert_embedding(fasta, tokenizer, model, max_length=1024, batch_size=32):
    dataset = SequenceDataset(fasta)
    dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, num_workers=4)

    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing FASTA embeddings", unit="batch"):

            tokenized = tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                  max_length=max_length, add_special_tokens=True)

            input_ids = tokenized["input_ids"].to(device)
            attention_mask = tokenized["attention_mask"].to(device)

            input_ids = input_ids.long()  # Convert to long tensor for BERT input
            attention_mask = attention_mask  # no necesita ser float16

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden_states = outputs.last_hidden_state

            # Mean pooling
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            all_embeddings.append(mean_pooled.cpu().numpy())

    return np.vstack(all_embeddings)


def smiles_to_chemberta_embedding(smiles, tokenizer, model, max_length=150):
    dataset = SmilesDataset(smiles)
    dataloader = DataLoader(dataset, batch_size=32)

    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing SMILES sequences", unit="batch"):
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            outputs = model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            attention_mask = inputs['attention_mask']

            input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
            sum_embeddings = torch.sum(last_hidden_states * input_mask_expanded, 1)
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            mean_pooled = sum_embeddings / sum_mask

            all_embeddings.append(mean_pooled.cpu().numpy())

    return np.vstack(all_embeddings)

class SequenceDataset(Dataset):
    """
    Dataset class for the FASTA sequences in order to preprocess them
    """

    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]
    
class SmilesDataset(Dataset):
    """
    Dataset class for the SMILES sequences in order to preprocess them
    """

    def __init__(self, smiles):
        self.smiles = smiles

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        return self.smiles[idx]

def preprocess_data(csv_path, tokenizer_fasta, model_fasta, tokenizer_smiles, model_smiles, save_path=None):
    df = pd.read_csv(csv_path)
    
    df = df[df['IC50'] <= 10000]
    
    tqdm.pandas()
    df['Smiles'] = df['Smiles'].progress_apply(lambda x: x if Chem.MolFromSmiles(x) else None)
    df['Sequences'] = df['Sequences'].progress_apply(lambda x: x if set(x).issubset(set("ACDEFGHIKLMNPQRSTVWY")) else None)
    df = df.dropna(subset=['Smiles', 'Sequences'])

    df['IC50'] = -np.log10(df['IC50']).astype(np.float32)

    
    # We use the IQR method to remove outliers from the IC50 values
    Q1 = df['IC50'].quantile(0.25)
    Q3 = df['IC50'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df = df[(df['IC50'] >= lower_bound) & (df['IC50'] <= upper_bound)]


    # Embedding the SMILES and FASTA sequences
    smiles_embeddings = smiles_to_chemberta_embedding(df['Smiles'].tolist(), tokenizer_smiles, model_smiles)
    df['smiles_embedding'] = list(smiles_embeddings)

    fasta_embeddings = fasta_to_bert_embedding(df['Sequences'].tolist(), tokenizer_fasta, model_fasta)
    df['bert_embedding'] = list(fasta_embeddings)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_pickle(save_path)
        print(f"Preprocessed data saved in {save_path}")
    
    return df

def load_preprocessed_data(pkl_path):
    if os.path.exists(pkl_path):
        print("Loading preprocessed data...")
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
    # By default, ChemBERTa and ESM2 embeddings are 384 and 320 dimensions respectively.
    def __init__(self, smiles_model_dim=768, fasta_model_dim=480, hidden_dim=512, dropout=0.3, num_layers=2, n_attention_heads=8, verbose=False):
        super(CombinedModel, self).__init__()
        
        self.fc_chem = nn.Sequential(
            nn.Linear(smiles_model_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers - 1)]
        )
        
        self.fc_bert = nn.Sequential(
            nn.Linear(fasta_model_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(num_layers - 1)]
        )

        if hidden_dim % n_attention_heads != 0:
            for i in range(n_attention_heads, 0, -1):
                if hidden_dim % i == 0:
                    n_attention_heads = i
                    if verbose:
                        print(f"Adjusted n_attention_heads to {n_attention_heads} to ensure divisibility with hidden_dim {hidden_dim}.")
                    break
            if hidden_dim % n_attention_heads != 0:
                n_attention_heads = 1
        
        self.bert_self_attention  = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_attention_heads, batch_first=True, dropout=dropout)
        self.chem_self_attention  = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_attention_heads, batch_first=True, dropout=dropout)

        self.cross_attention_chem_to_bert = nn.MultiheadAttention(hidden_dim, n_attention_heads, batch_first=True, dropout=dropout)
        self.cross_attention_bert_to_chem = nn.MultiheadAttention(hidden_dim, n_attention_heads, batch_first=True, dropout=dropout)


        self.norm_after_chem_attention = nn.LayerNorm(hidden_dim)
        self.norm_after_bert_attention = nn.LayerNorm(hidden_dim)
        self.norm_after_cross_chem = nn.LayerNorm(hidden_dim)
        self.norm_after_cross_bert = nn.LayerNorm(hidden_dim)
        
        self.fc_combined = nn.Sequential(
            # We use 3 hidden_dim because it's the hidden_dim of the smiles model, the fasta model and the product of both
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, chem_emb, bert_emb):
        chem_out = self.fc_chem(chem_emb)
        bert_out = self.fc_bert(bert_emb)

        chem_seq = chem_out.unsqueeze(1)
        bert_seq = bert_out.unsqueeze(1)

        sa_chem, _ = self.chem_self_attention(chem_seq, chem_seq, chem_seq)
        sa_bert, _ = self.bert_self_attention(bert_seq, bert_seq, bert_seq)

        chem_out_sa = self.norm_after_chem_attention(chem_out + sa_chem.squeeze(1))
        bert_out_sa = self.norm_after_bert_attention(bert_out + sa_bert.squeeze(1))

        chem_seq = chem_out_sa.unsqueeze(1)
        bert_seq = bert_out_sa.unsqueeze(1)

        cross_chem, _ = self.cross_attention_chem_to_bert(chem_seq, bert_seq, bert_seq)
        cross_bert, _ = self.cross_attention_bert_to_chem(bert_seq, chem_seq, chem_seq)

        chem_out_crossed = self.norm_after_cross_chem(chem_out_sa + cross_chem.squeeze(1))
        bert_out_crossed = self.norm_after_cross_bert(bert_out_sa + cross_bert.squeeze(1))

        element_wise_prod = chem_out_crossed * bert_out_crossed
        combined_features = torch.cat((chem_out_crossed, bert_out_crossed, element_wise_prod), dim=1)

        combined_out = self.fc_combined(combined_features)
        return self.output(combined_out)
    

def objective(trial, train_dataset, val_dataset):
    hidden_dim = trial.suggest_int("hidden_dim", 512, 1024)
    dropout = trial.suggest_float("dropout", 0.05, 0.2)
    lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True) 
    weight_decay = np.float32(trial.suggest_float("weight_decay", 1e-6, 1e-4))
    num_layers = trial.suggest_int("num_layers", 2, 4)  
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128]) 
    n_attention_heads = trial.suggest_int("n_attention_heads", 4, 12, step=2)


    smiles_embedding_dim = train_dataset.smiles_embeddings.shape[1]
    fasta_embedding_dim = train_dataset.bert_embeddings.shape[1]
    model = CombinedModel(smiles_model_dim=smiles_embedding_dim, fasta_model_dim=fasta_embedding_dim, hidden_dim=hidden_dim, dropout=dropout, num_layers=num_layers, n_attention_heads=n_attention_heads, verbose = args.verbose).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_model_with_optuna(trial, model, train_loader, val_loader, lr, weight_decay, epochs=50)

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_loss = float('inf')
        self.epochs_no_improve = 0
        self.early_stop = False

    def __call__(self, val_loss):
        current_loss = val_loss
        if current_loss < self.best_loss - self.delta:
            self.best_loss = current_loss
            self.epochs_no_improve = 0

        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.early_stop = True

def train_model_with_optuna(trial, model, train_loader, val_loader, lr, weight_decay, epochs=50):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler_patience = trial.suggest_int("scheduler_patience", 3, 7)
    scheduler_factor = trial.suggest_float("scheduler_factor", 0.3, 0.6, step=0.1) 
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=scheduler_factor, patience=scheduler_patience, verbose=False)
    criterion = nn.SmoothL1Loss()
    scaler = GradScaler()
    
    best_val_loss = float('inf')
    best_mse = float('inf')
    best_r2 = float('-inf')

    early_stopping_patience = trial.suggest_int("early_stopping_patience", 5, 15) 
    early_stopping_delta = trial.suggest_float("early_stopping_delta", 1e-4, 5e-3, log=True)
    early_stopping = EarlyStopping(patience=early_stopping_patience, delta=early_stopping_delta)
 


    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for chem_emb, bert_emb, target in train_loader:
            chem_emb, bert_emb, target = chem_emb.to(device), bert_emb.to(device), target.to(device)
            
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                output = model(chem_emb, bert_emb).squeeze(-1)
                loss = criterion(output, target)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
                
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
        avg_val_loss = val_loss / len(val_loader)

        r2 = r2_score(all_targets, all_preds)
        mse = mean_squared_error(all_targets, all_preds)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {avg_val_loss:.4f}, R2: {r2:.4f}, MSE: {mse:.4f}")

        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_r2 = r2
            best_mse = mse
        
            
        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    
    
    # Save model params in json format
    params = {
        "trial_number": trial.number,
        "hidden_dim": trial.params["hidden_dim"],
        "dropout": trial.params["dropout"],
        "lr": float(trial.params["lr"]),
        "weight_decay": float(trial.params["weight_decay"]),
        "num_layers": trial.params["num_layers"],
        "batch_size": trial.params["batch_size"],
        "n_attention_heads": trial.params["n_attention_heads"],
        "scheduler_patience": float(scheduler_patience),
        "scheduler_factor": float(scheduler_factor),
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_delta": float(early_stopping_delta),
        "best_val_loss": best_val_loss,
        "r2": best_r2,
        "mse": best_mse
    }
    all_trials = []

    if os.path.exists(f"{output_path}/all_trials.json"):
        if os.path.getsize(f"{output_path}/all_trials.json") > 0:
            with open(f"{output_path}/all_trials.json", "r") as f:
                all_trials = json.load(f)
         
    all_trials.append(params)
    with open(f"{output_path}/all_trials.json", "w") as f:
        json.dump(all_trials, f, indent=4)

    model_path = os.path.join(output_path, f"trial_{trial.number}_loss_{best_val_loss:.4f}.pth")
    with open(model_path, "wb") as f:
        torch.save(model.state_dict(), f)
    trial.set_user_attr("model_path", model_path)
    

    return best_val_loss

if __name__ == "__main__":
    torch._dynamo.config.suppress_errors = True
    torch._dynamo.disable()

    args = args_parser()

    if args.create_subfolders:
        output_path = os.path.join("models", "checkpoints", args.study_name)
        output_path = f"models/checkpoints/{args.study_name}"
    else:
        output_path = "models/checkpoints"
    os.makedirs(output_path, exist_ok=True)

    

    # We use esm2_t12_35M_UR50D and ChemBERTa-77M-MTR as the tokenizers and models for the embeddings
    print("Loading tokenizers and models...")
    tokenizer_fasta = AutoTokenizer.from_pretrained("facebook/esm2_t12_35M_UR50D",trust_remote_code=True)
    model_fasta = AutoModel.from_pretrained("facebook/esm2_t12_35M_UR50D", torch_dtype=torch.float16).to(device)
    tokenizer_smiles = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")
    model_smiles = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k").to(device)

    model_fasta = model_fasta.to(device).eval()
    model_smiles = model_smiles.to(device).eval()

    if hasattr(torch, "compile"):
        print("Compiling models for performance optimization...")
        model_fasta = torch.compile(model_fasta, backend="aot_eager")
        model_smiles = torch.compile(model_smiles, backend="aot_eager")
    
    df = load_preprocessed_data(args.preprocessed_data_path)
    if df is None:
        df = preprocess_data(args.raw_data_path, tokenizer_fasta, model_fasta, tokenizer_smiles, model_smiles, save_path=args.preprocessed_data_path)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = IC50Dataset(train_df)
    val_dataset = IC50Dataset(val_df)
    
    json_path = os.path.join("models/checkpoints", "all_trials.json")

    try:
        study = optuna.load_study(study_name=args.study_name, storage=args.storage)
        print("Study loaded.")
    except:
        print(f"No study found. Creating a new study: {args.study_name}")
        study = optuna.create_study(direction="minimize", study_name=args.study_name, storage=args.storage, pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1))
   

    
    if args.visualize:
        if not "sqlite:///" in args.storage:
            args.storage = f"sqlite:///{args.storage}"
        print("Launching Optuna dashboard...")
        #stdout=subprocess.DEVNULL to hide the output logs, which are constantly updating during the optimization.
        subprocess.Popen(f"optuna-dashboard {args.storage}", shell=True, stdout=subprocess.DEVNULL, creationflags=subprocess.CREATE_NEW_CONSOLE) 
        print("Optuna dashboard successfully launched. To access it, go to 'http://localhost:8080' in your browser")
    
    study.optimize(lambda trial: objective(trial, train_dataset, val_dataset), n_trials=args.n_trials)

    print(f"Study completed with {len(study.trials)} trials. Best value: {study.best_value}")

    for file in os.listdir(output_path):
        if file.endswith(".pth"):
            if study.best_value != float(file.split("_")[-1].split(".")[0]):
                os.remove(os.path.join(output_path, file))
            else:
                print(f"Best model saved in {os.path.join(output_path, file)} with value {study.best_value}")

                



    
