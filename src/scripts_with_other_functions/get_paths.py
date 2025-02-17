import json
import os
import numpy as np

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

directory = "data/smiles"
file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file))]

unique_chars_dict = create_unique_chars_dict(file_paths)
save_unique_chars_dict(unique_chars_dict, "models/test_unique_chars_dict.json")
print(unique_chars_dict)