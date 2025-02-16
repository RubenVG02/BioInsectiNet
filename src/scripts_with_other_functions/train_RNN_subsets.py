import argparse
import os
import json
import subprocess

def load_or_initialize_epoch_log(models_dir, subsets, log_file):
    log_path = os.path.join(models_dir, log_file)
    
    if not os.path.exists(log_path):
        epoch_log = {subset: 0 for subset in subsets}
        with open(log_path, "w") as f:
            json.dump(epoch_log, f, indent=4)
    
    with open(log_path, "r") as f:
        epoch_log = json.load(f)

    return epoch_log


def train_subsets(subset, models_dir, epochs, batch_size, learning_rate, log_file):
    print(f"Training model for subset {subset}...")
    subprocess.run([
        "python", "src/train_RNN_generation.py",
        "--data_path", f"{models_dir}/{subset}.txt",
        "--output_dir", models_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(learning_rate),
        "--patience", "5",
        "--json_epochs_path", f"{models_dir}/{log_file}",
    ])

    print(f"Model for subset {subset} trained successfully!")

def main():
    parser = argparse.ArgumentParser(description="Train RNN models for different subsets of SMILES data.")
    parser.add_argument("models_dir", type=str, help="Directory to save trained models.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training.")
    parser.add_argument("--log_file", type=str, default="epoch_log.json", help="File to save the epoch log.")
    args = parser.parse_args()

    subsets = os.listdir(args.models_dir)
    epoch_log = load_or_initialize_epoch_log(args.models_dir, subsets, args.log_file)

    for subset in subsets:
        train_subsets([subset], args.models_dir, args.epochs - epoch_log[subset], args.batch_size, args.learning_rate, args.log_file)

    print("All models trained successfully!")