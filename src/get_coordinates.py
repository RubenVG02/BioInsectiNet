import os
import subprocess
import argparse
import requests
from rdkit import Chem
from rdkit.Chem import AllChem
import pymol
import numpy as np
import matplotlib.pyplot as plt
import platform
import stat


def parse_arguments():
    parser = argparse.ArgumentParser(description="Docking pipeline using RDKit + Meeko + Vina.")
    parser.add_argument("--smile", type=str, required=True, help="SMILES of the ligand")
    parser.add_argument("--fasta", type=str, required=True, help="FASTA sequence of the receptor")
    parser.add_argument("--center", type=float, nargs=3, default=[0,0,0], help="Center of docking box (x,y,z)")
    parser.add_argument("--box_size", type=float, nargs=3, default=[20,20,20], help="Size of docking box (x,y,z)")
    parser.add_argument("--output_dir", type=str, default="output_vina", help="Output directory")
    parser.add_argument("--vina_path", type=str, default=r"./src/vina.exe", help="Path to Vina executable if not in PATH")
    return parser.parse_args()

def detect_os():
    """Detect the operating system to adjust paths and commands if necessary."""
    os_name = platform.system().lower()
    if os_name == 'windows':
        return 'windows'
    elif os_name == 'linux':
        return 'linux'
    elif os_name == 'darwin':
        return 'macos'
    else:
        raise RuntimeError(f"Unsupported OS: {os_name}")

def fasta_to_pdb_esmfold(fasta_sequence, output_dir):
    url = "https://api.esmatlas.com/foldSequence/v1/pdb/"
    response = requests.post(url, data=fasta_sequence)
    if response.status_code != 200:
        raise Exception(f"ESMFold API error: {response.status_code} {response.text}")
    pdb_file = os.path.join(output_dir, "receptor.pdb")
    with open(pdb_file, "w") as f:
        f.write(response.text)
    print(f"[INFO] receptor.pdb downloaded and saved to {pdb_file}")
    return pdb_file

def prepare_ligand_meeko(smiles, output_pdbqt):
    try:
        from meeko import MoleculePreparation, PDBQTMolecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Could not read SMILES of ligand")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
             
        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        if not mol_setups:
            raise RuntimeError("Meeko failed to prepare ligand")
        pdbqt_string = PDBQTMolecule(mol_setups[0], only_first=True).write_string()
        
        with open(output_pdbqt, "w") as f:
            f.write(pdbqt_string)
        print(f"[INFO] Ligand prepared with Meeko and saved to {output_pdbqt}")
        return True
    except Exception as e:
        print(f"[ERROR] Meeko ligand preparation failed: {e}")
        return False


def prepare_ligand_obabel(smiles, output_pdbqt):
    """Alternative ligand preparation using Open Babel"""
    try:
        # First convert SMILES to PDB using RDKit
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Could not read SMILES of ligand")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        
        pdb_file = output_pdbqt.replace('.pdbqt', '.pdb')
        Chem.MolToPDBFile(mol, pdb_file)
        
        # Then convert PDB to PDBQT using obabel
        command = ["obabel", pdb_file, "-O", output_pdbqt, "--partialcharge", "gasteiger"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Open Babel error: {result.stderr}")
        
        print(f"[INFO] Ligand prepared with Open Babel and saved to {output_pdbqt}")
        return True
    except Exception as e:
        print(f"[ERROR] Open Babel ligand preparation failed: {e}")
        return False

def prepare_receptor_meeko(input_pdb, output_pdbqt):
    try:
        from meeko import MoleculePreparation, PDBQTMolecule
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromPDBFile(input_pdb, removeHs=False)
        if mol is None:
            raise ValueError("Could not load receptor PDB file")

        mol = Chem.AddHs(mol, addCoords=True)

        if mol.GetNumConformers() == 0:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        AllChem.UFFOptimizeMolecule(mol)

        Chem.MolToPDBFile(mol, input_pdb)

        mol = Chem.MolFromPDBFile(input_pdb, removeHs=False)
        if mol is None:
            raise ValueError("Could not load receptor PDB file after preprocessing")
        
        if mol.GetNumAtoms() == 0:
            raise ValueError("Receptor PDB file contains no atoms")

        preparator = MoleculePreparation(hydrate=False, rigid_macrocycles=True)
        setups = preparator.prepare(mol)
        
        if not setups:
            raise RuntimeError("Meeko failed to prepare receptor - no setups generated")
        
        pdbqt_string = PDBQTMolecule(setups[0], only_first=True).write_string()
        
        with open(output_pdbqt, "w") as f:
            f.write(pdbqt_string)
        print(f"[INFO] Receptor prepared with Meeko and saved to {output_pdbqt}")
        return True
    except Exception as e:
        print(f"[ERROR] Meeko receptor preparation failed: {e}")
        return False

def prepare_receptor_adt(input_pdb, output_pdbqt):
    """Receptor preparation using AutoDockTools"""
    try:
        command = ["prepare_receptor", "-r", input_pdb, "-o", output_pdbqt]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"AutoDockTools error: {result.stderr}")
        print(f"[INFO] Receptor prepared with AutoDockTools and saved to {output_pdbqt}")
        return True
    except Exception as e:
        print(f"[ERROR] AutoDockTools receptor preparation failed: {e}")
        return False

def prepare_receptor_obabel(input_pdb, output_pdbqt):
    """Receptor preparation using Open Babel"""
    try:
        command = ["obabel", input_pdb, "-O", output_pdbqt, "-xr"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Open Babel error: {result.stderr}")
        print(f"[INFO] Receptor prepared with Open Babel and saved to {output_pdbqt}")
        return True
    except Exception as e:
        print(f"[ERROR] Open Babel receptor preparation failed: {e}")
        return False

def run_docking_vina(ligand_pdbqt, receptor_pdbqt, center, box_size, output_dir, vina_path):
    out_docked = os.path.join(output_dir, "docked.pdbqt")

    vina_paths = {
        "linux":"vina_1.2.7_linux_x86_64",
        "darwin":"vina_1.2.7_mac_x86_64",
        "windows":"vina_1.2.7_win.exe"
    }
    base_download_url = "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.7/"
    if not os.path.exists(vina_path):
        os_name = detect_os()
        print(f"[INFO] Detected OS: {os_name}. Downloading Vina executable...")
        for vina_key, vina_exe in vina_paths.items():
            if vina_key == os_name:
                download_url = base_download_url + vina_exe
                response = requests.get(download_url)
                print(f"[DEBUG] Downloading Vina from {download_url}")
                if response.status_code == 200:
                    with open(vina_path, "wb") as f:
                        f.write(response.content)
                    st = os.stat(vina_path)
                    os.chmod(vina_path, st.st_mode | stat.S_IEXEC)  # Make executable
                    print(f"[INFO] Vina executable downloaded and saved to {vina_path}")
                    break
        
    command = [
        vina_path,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--center_x", str(center[0]),
        "--center_y", str(center[1]),
        "--center_z", str(center[2]),
        "--size_x", str(box_size[0]),
        "--size_y", str(box_size[1]),
        "--size_z", str(box_size[2]),
        "--out", out_docked,
        "--num_modes", "9",
        "--seed", "42",
        "--exhaustiveness", "8",
    ]
    print(f"[DEBUG] Running Vina: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Vina error: {result.stderr}")
    print(f"[INFO] Docking completed. Results in {out_docked}")

def visualize_docking_results(docked_pdbqt, receptor_pdbqt, output_dir):
    """Visualize docking results using PyMOL or similar tool"""
    try:
        pymol.cmd.reinitialize()  # Clear previous state
        pymol.cmd.load(docked_pdbqt, "docked")
        pymol.cmd.load(receptor_pdbqt, "receptor")
        pymol.cmd.show("sticks", "docked")
        pymol.cmd.show("cartoon", "receptor")
        pymol.cmd.save(os.path.join(output_dir, "docking_results.pse"))
        print(f"[INFO] Docking results visualized and saved to {output_dir}/docking_results.pse")
    except ImportError:
        print("[WARNING] PyMOL not available for visualization. Skipping this step.")

def extract_affinities_from_vina_output(output_file):
    with open(output_file, "r") as f:
        affinities = []
        for line in f:
            if line.startswith("REMARK VINA RESULT:"):
                parts = line.strip().split()
                try:
                    affinity = float(parts[3])
                    affinities.append(affinity)
                except (IndexError, ValueError):
                    print(f"[WARNING] Could not parse affinity from line: {line.strip()}")
    if not affinities:
        print("[WARNING] No valid affinities found.")
    return affinities

def compute_affinity_statistics(affinities):
    if not affinities:
        raise ValueError("No affinities found for statistics calculation.")
    affinities_np = np.array(affinities)
    stats = {
        "mean": np.mean(affinities_np),
        "std": np.std(affinities_np),
        "min": np.min(affinities_np),
        "max": np.max(affinities_np),
        "median": np.median(affinities_np),
    }
    return stats

def plot_affinity_histogram(affinities, output_file):
    """Plot histogram of binding affinities"""
    if not affinities:
        print("[WARNING] No affinities to plot.")
        return
    bins = np.linspace(min(affinities), max(affinities), 100)
    plt.hist(affinities, bins=bins, color='blue', alpha=0.7)
    plt.xlabel('Binding Affinity (kcal/mol)')
    plt.ylabel('Frequency')
    plt.title('Binding Affinity Distribution')
    plt.grid(axis='y', alpha=0.75, linestyle='--')
    plt.savefig(output_file)
    print(f"[INFO] Affinity histogram saved to {output_file}")

def viability_check(min_affinity, max_affinity, stats):
    """Check if docking results are viable based on affinity statistics"""
    if not stats:
        print("[WARNING] No affinity statistics found.")
        return
    if stats["min"] < min_affinity and stats["max"] > max_affinity:
        print("[INFO] Docking likely viable: good affinity found.")
        return True
    else:
        print("[WARNING] Docking affinities are not very strong, consider reviewing setup or ligand suitability.")
        return False
    

def main():
    args = parse_arguments()
    if os.path.exists(args.output_dir):
        i = 1
        while os.path.exists(f"{args.output_dir}_{i}"):
            i += 1
        args.output_dir = f"{args.output_dir}_{i}"
        print(f"[INFO] Output directory set to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Generating receptor PDB from FASTA using ESMFold...")
    pdb_file = fasta_to_pdb_esmfold(args.fasta, args.output_dir)

    print("[INFO] Preparing receptor PDBQT...")
    receptor_pdbqt = os.path.join(args.output_dir, "receptor.pdbqt")
    
    # Try different methods in order of preference
    methods_tried = []
    
    # Try Meeko first
    if not prepare_receptor_meeko(pdb_file, receptor_pdbqt):
        methods_tried.append("Meeko")
        # Try Open Babel as last resort
        if not prepare_receptor_obabel(pdb_file, receptor_pdbqt):
            methods_tried.append("Open Babel")
            raise RuntimeError(f"Failed to prepare receptor with all methods tried: {', '.join(methods_tried)}")

    print("[INFO] Preparing ligand PDBQT...")
    ligand_pdbqt = os.path.join(args.output_dir, "ligand.pdbqt")
    
    # Try different methods for ligand
    ligand_methods_tried = []
    
    # Try Meeko first
    if not prepare_ligand_meeko(args.smile, ligand_pdbqt):
        ligand_methods_tried.append("Meeko")
        # Try Open Babel
        if not prepare_ligand_obabel(args.smile, ligand_pdbqt):
            ligand_methods_tried.append("Open Babel")
            raise RuntimeError(f"Failed to prepare ligand with all methods tried: {', '.join(ligand_methods_tried)}")

    print("[INFO] Running docking with Vina...")
    run_docking_vina(ligand_pdbqt, receptor_pdbqt, args.center, args.box_size, args.output_dir, args.vina_path)

    print(f"[INFO] Visualizing docking results in PyMOL...")
    visualize_docking_results(os.path.join(args.output_dir, "docked.pdbqt"), receptor_pdbqt, args.output_dir)

    print("[INFO] Extracting binding affinities from docking output...")
    output_file = os.path.join(args.output_dir, "docked.pdbqt")
    affinities = extract_affinities_from_vina_output(output_file)
    if not affinities:
        print("[WARNING] No binding affinities found in docking output.")
    else:
        plot_affinity_histogram(affinities, os.path.join(args.output_dir, "affinity_histogram.png"))
        stats = compute_affinity_statistics(affinities)
    viability_check(stats=stats, min_affinity=-7, max_affinity=0.0)


    print("[INFO] Pipeline completed successfully.")
    print(f"[INFO] Output files in {args.output_dir}")
    pse_file = os.path.join(args.output_dir, "docking_results.pse")
    if os.path.exists(pse_file):
        print(f"[INFO] Opening PyMOL with results: {pse_file}")
        pymol.cmd.reinitialize()  # Clear previous state
        subprocess.run(["pymol", pse_file])
    

if __name__ == "__main__":
    main()