import json
import os
import numpy as np
import argparse

PAD_TOKEN = "<PAD>"

def compute_unique_chars(smiles_list):
    return sorted(set("".join(smiles_list)) | {'\n', PAD_TOKEN})

def get_90th_percentile_length(smiles_list):
    percentile_90 = int(np.percentile([len(smiles) for smiles in smiles_list], 90))
    print(f"90th percentile length: {percentile_90}")
    return percentile_90

def truncate_data_90_percentile(smiles_list, max_length):
    return [smiles[:max_length] for smiles in smiles_list]

def create_unique_chars_dict(file_paths):
    unique_chars_dict = {}
    for file_path in file_paths:
        with open(file_path, "r") as f:
            smiles_list = f.read().splitlines()
        
        print("Processing", file_path)
        percentile_90 = get_90th_percentile_length(smiles_list)
        print(percentile_90)
        
        # If the 90th percentile is greater than 155, truncate the data to that length
        if percentile_90 > 155:
            print("Truncating data to 90th percentile length.")
            smiles_list = truncate_data_90_percentile(smiles_list, 155) # Truncate to 155 characters
        
        unique_chars = compute_unique_chars(smiles_list)
        unique_chars_dict[file_path] = unique_chars
    
    return unique_chars_dict

def save_unique_chars_dict(unique_chars_dict, output_file):
    with open(output_file, "w") as f:
        json.dump(unique_chars_dict, f)

def load_unique_chars_dict(input_file):
    with open(input_file, "r") as f:
        return json.load(f)

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def main():
    parser = argparse.ArgumentParser(description="Process SMILES files and compute unique characters.")
    parser.add_argument("--input_dir", type=str, default="data/smiles", help="Directory containing SMILES files. Each file should contain SMILES strings separated by newlines.")
    parser.add_argument("--output_file", type=str, default= "models/unique_chars_dict.json", help="Output JSON file to save unique characters dictionary. Default: unique_chars_dict.json in the models directory.")
    args = parser.parse_args()

    ensure_directory_exists(args.input_dir)

    file_paths = []
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            file_paths.append(os.path.join(root, file))
        if not file_paths:
            print(f"No files found in the directory: {args.input_dir}")
            return

    unique_chars_dict = create_unique_chars_dict(file_paths)

    output_dir = os.path.dirname(args.output_file)
    ensure_directory_exists(output_dir)

    save_unique_chars_dict(unique_chars_dict, args.output_file)
    print(f"Unique characters dictionary saved to: {args.output_file}")

if __name__ == "__main__":
    main()