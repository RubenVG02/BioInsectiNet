import argparse
import numpy as np
import multiprocessing
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Split a large SMILES dataset into balanced subsets.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the input .txt file")
    parser.add_argument("--num_subsets", type=int, default=10, help="Number of subsets to create (default: 10)")
    parser.add_argument("--subset_size", type=int, default=250000, help="Size of each subset (default: 250000)")
    parser.add_argument("--clusters", type=int, default=200, help="Number of clusters for diversity balancing. More clusters, more complexity (default: 200)")
    parser.add_argument("--batch_size", type=int, default=10000, help="Batch size for clustering (default: 10000)")
    parser.add_argument("--n_bits", type=int, default=1024, help="Number of bits for Morgan fingerprints (default: 1024)")
    return parser.parse_args()

def load_smiles(file_path, num_subsets, subset_size):
    with open(file_path, "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]

        if len(smiles_list) <= num_subsets * subset_size:
            raise ValueError("Dataset is too small for the specified number of subsets and subset size. Please use a larger dataset or reduce the number of subsets and/or subset size.")
    return smiles_list

def smiles_to_fingerprint(smiles, radius=2, n_bits=1024):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits), dtype=np.uint8)
    return None

def compute_fingerprints(smiles_list, n_bits):
    with multiprocessing.Pool() as pool:
        fingerprints = list(tqdm(pool.imap(smiles_to_fingerprint, smiles_list, n_bits), total=len(smiles_list), desc="Computing fingerprints"))
    valid_data = [(smi, fp) for smi, fp in zip(smiles_list, fingerprints) if fp is not None]
    return valid_data

def cluster_smiles(fingerprints, n_clusters, batch_size):
    X = np.vstack([fp for _, fp in fingerprints]) 
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=batch_size, n_init=5)

    num_batches = (len(X) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(X), batch_size), total=num_batches, desc="Training MiniBatchKMeans"):
        batch = X[i:i + batch_size]  
        kmeans.partial_fit(batch)

    cluster_labels = kmeans.predict(X)
    return [(smiles, cluster) for (smiles, _), cluster in zip(fingerprints, cluster_labels)]

def stratified_sampling(clustered_data, num_subsets, subset_size):
    from collections import defaultdict
    subsets = [[] for _ in range(num_subsets)]
    cluster_dict = defaultdict(list)

    for smiles, cluster in clustered_data:
        cluster_dict[cluster].append(smiles)

    total_smiles = num_subsets * subset_size

    for cluster, smiles_list in tqdm(cluster_dict.items(), desc="Distributing clusters"):
        np.random.shuffle(smiles_list)
        per_subset = min(len(smiles_list) // num_subsets, subset_size)
        for i in range(num_subsets):
            subsets[i].extend(smiles_list[i * per_subset: (i + 1) * per_subset])

    final_subsets = [subset[:subset_size] for subset in subsets]
    return final_subsets

def save_subsets(subsets, name_input):
    os.makedirs(f"data/smiles/{name_input}_subsets", exist_ok=True)
    for i, subset in enumerate(subsets):
        filename = f"data/smiles/{name_input}_subsets/{name_input}_subset_{i+1}.txt"
        with open(filename, "w") as f:
            f.writelines(smi + "\n" for smi in subset)

def main():
    args = parse_args()
    name_dir = os.path.basename(args.file_path)
    name_dir = name_dir.replace(".txt", "").replace(".", "_")

    print("Loading dataset...")
    smiles_list = load_smiles(args.file_path, args.num_subsets, args.subset_size)

    print("Computing fingerprints in parallel...")
    fingerprints = compute_fingerprints(smiles_list, args.n_bits)

    print("Applying clustering...")
    clustered_data = cluster_smiles(fingerprints, args.clusters, args.batch_size)

    print("Generating balanced subsets...")
    subsets = stratified_sampling(clustered_data, args.num_subsets, args.subset_size)

    print("Saving subsets...")
    save_subsets(subsets, name_dir)

    print("Process completed!")

if __name__ == "__main__":
    main()
