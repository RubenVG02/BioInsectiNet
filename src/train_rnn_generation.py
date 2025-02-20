import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import argparse
import json

# Silence rdkit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train an RNN model to generate SMILES sequences.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the SMILES dataset.")
    parser.add_argument("--output_dir", type=str, default="models/generator", help="Directory to save the trained model. By default, it saves the model in the 'models/generator' directory.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train the model. By default, it trains the model for 100 epochs.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training the model. By default, it uses a batch size of 128.")
    parser.add_argument("--embedding_dim", type=int, default=256, help="Dimension of the embedding layer. By default, it uses an embedding dimension of 256.")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of the LSTM layer. By default, it uses a hidden size of 256.")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of layers in the LSTM. By default, it uses 3 layers.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for training the model. By default, it uses a learning rate of 1e-3.")
    parser.add_argument("--patience", type=int, default=5, help="Number of epochs to wait before early stopping. By default, it uses a patience of 5.")
    parser.add_argument("--log", action="store_true", help="If set, logs will be saved to the default log_path ('training_epochs.json').")
    parser.add_argument("--log_path", type=str, default="training_epochs.json", help="JSON file to save the training epochs. Useful for subset training")
    parser.add_argument("--custom_name", type=str, default=None, help="Custom name for the model file. By default, it uses the file name of the dataset.")
    parser.add_argument("--filter_percentile", type=bool, default=True, help="Filter the dataset to the 90th percentile length.")
    return parser.parse_args()

def load_smiles(file_path):
    with open(file_path, "r") as f:
        smiles = [line.strip() for line in f.readlines()]
    return smiles

class SMILESDataset(Dataset):
    def __init__(self, smiles, char_to_idx, max_length):
        self.smiles = smiles
        self.char_to_idx = char_to_idx
        self.max_length = max_length

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        seq = self.smiles[idx]
        encoded = [self.char_to_idx[char] for char in seq if char in self.char_to_idx]
        
        # We add a newline character at the end of the sequence
        encoded.append(self.char_to_idx['\n'])
        
        # Fill the sequence with padding tokens if it's shorter than max_length
        if len(encoded) < self.max_length:
            encoded += [self.char_to_idx[PAD_TOKEN]] * (self.max_length - len(encoded))
        else:
            encoded = encoded[:self.max_length]  # Truncate the sequence if it's longer than max_length
        
        inputs = torch.tensor(encoded[:-1], dtype=torch.long)
        targets = torch.tensor(encoded[1:], dtype=torch.long) # Target is the next character in the sequence to predict
        lengths = torch.tensor(len(encoded) - 1, dtype=torch.long)
        
        return inputs, targets, lengths

def collate_fn(batch, char_to_idx):
    inputs, targets, lengths = zip(*batch)
    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=char_to_idx[PAD_TOKEN])
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=char_to_idx[PAD_TOKEN])
    lengths = torch.tensor(lengths, dtype=torch.long)
    return inputs, targets, lengths

class ImprovedSMILESGenerator(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, char_to_idx):
        super(ImprovedSMILESGenerator, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.char_to_idx = char_to_idx  

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=char_to_idx[PAD_TOKEN])
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, lengths, hidden=None):
        x = self.embedding(x)
        x = self.dropout_layer(x)
        
        # We pack the sequence to avoid unnecessary computations on padding tokens
        x = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        output, hidden = self.lstm(x, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True, total_length=max(lengths))

        output = self.dropout_layer(output)
        output = self.fc(output)
        return output, hidden
    
def get_percentile_95(smiles_list):
    lengths = [len(smile) for smile in smiles_list]
    percentile_90 = np.percentile(lengths, 95)
    print(f"90th percentile: {percentile_90}")
    return percentile_90


def save_epoch_state(json_epochs_path, subset, current_epoch, early_stop=False):
    epoch_state = {}
    if os.path.exists(json_epochs_path):
        with open(json_epochs_path, "r") as f:
            epoch_state = json.load(f)
    
    if early_stop:
        epoch_state[subset] = (current_epoch, True)
    else:
        epoch_state[subset] = (current_epoch, False)
    
    with open(json_epochs_path, "w") as f:
        json.dump(epoch_state, f, indent=4)

def train_model():
    if not args.custom_name:
        file_name = os.path.basename(args.file_path).split(".")[0]
    else:
        file_name = args.custom_name
    print("Loading data and tokenizing...")
    smiles_list = load_smiles(args.file_path)
    print(max(len(s) for s in smiles_list))
    if args.filter_percentile:
        if max(len(s) for s in smiles_list) < 160:
            max_length = max(len(s) for s in smiles_list) + 1  # +1 to account for newline character
            print(f"Max length: {max_length}")
        else:
            max_length = int(get_percentile_95(smiles_list) + 1)  # using 90th percentile to set the max_length instead of the longest sequence (mainly used for the longest sequence dataset)
            print("Max length set to 90th percentile." + str(max_length)) 
    else:
        max_length = max(len(s) for s in smiles_list) + 1  # +1 to account for newline character
        print(f"Max length: {max_length}")

    

    unique_chars = sorted(set("".join(smiles_list)) | {'\n', PAD_TOKEN})  # Include newline and padding token
    
    char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    
    all_chars_in_smiles = set("".join(smiles_list))
    vocab_chars = set(char_to_idx.keys())
    unknown_chars = all_chars_in_smiles - vocab_chars
    if unknown_chars:
        raise ValueError("Unknown characters in SMILES: {}".format(unknown_chars))

    print("Creating dataset and dataloader...")
    
    
    train_smiles, val_smiles = train_test_split(smiles_list, test_size=0.2, random_state=42)
    train_dataset = SMILESDataset(train_smiles, char_to_idx, max_length)
    val_dataset = SMILESDataset(val_smiles, char_to_idx, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              collate_fn=lambda x: collate_fn(x, char_to_idx))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                            collate_fn=lambda x: collate_fn(x, char_to_idx))

    model = ImprovedSMILESGenerator(len(char_to_idx), args.embedding_dim, args.hidden_size , args.num_layers, DROPOUT, char_to_idx)
    criterion = nn.CrossEntropyLoss(ignore_index=char_to_idx[PAD_TOKEN])  
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Verify if there are pre-trained models
    model_dir = args.output_dir
    model_pattern = os.path.join(model_dir, f"{file_name}_*.pth")  
    model_files = glob.glob(model_pattern)  

    if model_files:
        latest_model_path = max(model_files, key=os.path.getctime)
        print(f"Loading pre-trained model from {latest_model_path}...")
        model.load_state_dict(torch.load(latest_model_path, map_location=device))
        print("Pre-trained model loaded successfully!")
    else:
        print("No pre-trained model found. Starting from scratch.")
        

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    print("Training model...")
    smoother = SmoothingFunction().method1  # Function to smooth the BLEU score
    loss_scores = []
    bleu_scores = []
    val_loss_scores = []
    val_bleu_scores = []
    min_loss = float("inf")
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_bleu = 0
        num_batches = len(train_loader)

        with tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}", unit="batch") as tepoch:
            for inputs, targets, lengths in tepoch:
                inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
                optimizer.zero_grad()

                outputs, _ = model(inputs, lengths)
                loss = criterion(outputs.view(-1, len(char_to_idx)), targets.view(-1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)  # Gradient clipping
                optimizer.step()

                total_loss += loss.item()

                bleu_scores_batch = []
                predicted = torch.argmax(outputs, dim=-1).cpu().numpy()
                targets_cpu = targets.cpu().numpy()
                for pred, tgt in zip(predicted, targets_cpu):
                    pred_seq = [idx_to_char[idx] for idx in pred if idx != char_to_idx[PAD_TOKEN]]  
                    tgt_seq = [idx_to_char[idx] for idx in tgt if idx != char_to_idx[PAD_TOKEN]]   
                    bleu_scores_batch.append(sentence_bleu([tgt_seq], pred_seq, smoothing_function=smoother))
                batch_bleu = sum(bleu_scores_batch) / len(bleu_scores_batch)
                total_bleu += batch_bleu

                tepoch.set_postfix(loss=loss.item(), bleu=batch_bleu)

        avg_loss = total_loss / num_batches
        avg_bleu = total_bleu / num_batches
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.4f}, BLEU Score: {avg_bleu:.4f}")

        model.eval()
        val_loss = 0
        val_bleu = 0
        with torch.no_grad():
            with tqdm(val_loader, desc="Validation", unit="batch") as tval:
                for inputs, targets, lengths in tval:
                    inputs, targets, lengths = inputs.to(device), targets.to(device), lengths.to(device)
                    outputs, _ = model(inputs, lengths)
                    loss = criterion(outputs.view(-1, len(char_to_idx)), targets.view(-1))
                    val_loss += loss.item()

                    predicted = torch.argmax(outputs, dim=-1).cpu().numpy()
                    targets_cpu = targets.cpu().numpy()
                    for pred, tgt in zip(predicted, targets_cpu):
                        pred_seq = [idx_to_char[idx] for idx in pred if idx != char_to_idx[PAD_TOKEN]]
                        tgt_seq = [idx_to_char[idx] for idx in tgt if idx != char_to_idx[PAD_TOKEN]]
                        val_bleu += sentence_bleu([tgt_seq], pred_seq, smoothing_function=smoother)

                    tval.set_postfix(loss=loss.item())

        val_loss /= len(val_loader)
        val_bleu /= len(val_loader.dataset)
        print(f"Validation Loss: {val_loss:.4f}, Validation BLEU Score: {val_bleu:.4f}")

        scheduler.step(val_loss)
        loss_scores.append(avg_loss)
        bleu_scores.append(avg_bleu)
        val_loss_scores.append(val_loss)
        val_bleu_scores.append(val_bleu)

        # Early Stopping
        if val_loss < min_loss:
            min_loss = val_loss
            patience_counter = 0
            print("Model Improved! Saving...")
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(model.state_dict(), os.path.join(model_dir, f"{file_name}_{curr_date}.pth"))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:

                print("Early stopping triggered.")
                if args.log_path is not None:
                    print("Saving epoch state...")
                    if not "/" in args.log_path:
                        args.log_path = os.path.join(model_dir, args.log_path)
                    save_epoch_state(args.log_path, file_name, epoch + 1, early_stop=True)
                break

        if args.log:
            print("Saving epoch state...")
            if not "/" in args.log_path:
                args.log_path = os.path.join(model_dir, args.log_path)
            save_epoch_state(args.log_path, file_name, epoch + 1)

    print("Training finished!")

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 2, 1)
    plt.plot(loss_scores, color='red', label='Train Loss')
    plt.plot(val_loss_scores, color='orange', label='Validation Loss')
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(bleu_scores, color='blue', label='Train BLEU')
    plt.plot(val_bleu_scores, color='green', label='Validation BLEU')
    plt.title("BLEU Score")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU Score")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, f"metrics_{file_name}_{curr_date}.png"))

if __name__ == "__main__":
    args = parse_arguments()

    curr_date = str(time.time()).split(".")[0]
    DROPOUT = 0.2
    PAD_TOKEN = "<PAD>"  

    train_model()
    print("Done!")