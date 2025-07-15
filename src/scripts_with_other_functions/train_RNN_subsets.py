import argparse
import os
import json
import subprocess

import tkinter as tk
from tkinter import filedialog
def load_or_initialize_epoch_log(subsets, log_file):
    log_path = os.path.join(log_file)
    
    if not os.path.exists(log_path):
        subsets = [subset.replace(".txt", "") for subset in subsets]
        epoch_log = {subset: (0, False) for subset in subsets}  # 0 for the epochs trained, False for the early stopping flag
        with open(log_path, "w") as f:
            json.dump(epoch_log, f, indent=4)
        
        print(f"Epoch log file created at {log_path}")
    
    with open(log_path, "r") as f:
        epoch_log = json.load(f)

    return epoch_log


def train_subsets(subset, models_dir, epochs, batch_size, learning_rate, log_file, data_dir):


    file_path = f"{data_dir}/{subset}.txt"

    print(f"Training model for subset {subset}...")

    subprocess.run([
        "python", "src/train_RNN_generation.py",
        "--file_path", file_path,
        "--output_dir", models_dir,
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--lr", str(learning_rate),
        "--patience", "5",
        "--log",
        "--log_path", f"{log_file}",
        "--filter_percentile", "True",


    ])

    print(f"[INFO] Model for subset {subset} trained successfully!")

def select_dir():
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title="Select directory with SMILES subsets")
    if not directory:
        raise ValueError("You must select a directory.")
    return directory


def main():
    parser = argparse.ArgumentParser(description="Train RNN models for different subsets of SMILES data.")
    parser.add_argument("data_dir", type=str, help="Directory containing the SMILES subsets. Each subset should be a .txt file with the SMILES strings separated by newlines.")
    parser.add_argument("--models_dir", type=str, help="Directory to save trained models. By default, it selects models/generator/{os.path.basename(args.data_dir)}")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs for training. By default, 50")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training. By default, 128")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training. By default, 0,001")
    parser.add_argument("--log_path", type=str, default="epoch_log.json", help="File to save the epoch log. By default, epoch_log.json inside the models_dir")
    args = parser.parse_args()

    if args.data_dir is None or not os.path.exists(args.data_dir):
        print("[WARNING] No data directory specified or the directory does not exist. Please select a valid directory.")
        args.data_dir = select_dir()
        print(f"[INFO] Selected directory: {args.data_dir}")

    
    subsets = os.listdir(args.data_dir)


    if args.models_dir is None:
        print(f"[WARNING] No models directory specified. Using default models/generator/{os.path.basename(args.data_dir)}")
        args.models_dir = f"models/generator/{os.path.basename(args.data_dir)}"
    os.makedirs(args.models_dir, exist_ok=True)
    print(f"[INFO] Models will be saved in: {args.models_dir}")

    if not "/" in args.log_path: # If the log path is not a full or relative path, save it in the models_dir
        args.log_file = os.path.join(args.models_dir, args.log_path)
    epoch_log = load_or_initialize_epoch_log(subsets, args.log_file)

    print(f"[INFO] Epoch log file: {args.log_file}")
    print(f"[INFO] Epoch log loaded with {len(epoch_log)} subsets.")
    for subset in subsets:
        subset = subset.replace(".txt", "")
        if epoch_log[subset][0] >= args.epochs or epoch_log[subset][1] == True: # If the model has already been trained for the specified number of epochs or if early stopping was triggered (therefore, the model is already trained).
            print(f"[WARNING] Skipping training for subset {subset}. Already trained for {epoch_log[subset][0]} epochs or early stopping was triggered.")
            continue
        train_subsets(subset, args.models_dir, args.epochs - epoch_log[subset][0], args.batch_size, args.learning_rate, args.log_file, args.data_dir)

    print("[INFO] All subsets processed. Training complete.")
    
if __name__ == "__main__":
    main()