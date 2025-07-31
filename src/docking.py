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

from utils.simple_logger import log_info, log_warning, log_error, log_success

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for RDKit on some systems



def parse_arguments():
    parser = argparse.ArgumentParser(description="Docking pipeline using RDKit + Meeko + Vina.")
    parser.add_argument("--smile", type=str, required=True, help="SMILES of the ligand")
    parser.add_argument("--fasta", type=str, required=True, help="FASTA sequence of the receptor")
    parser.add_argument("--center", type=float, nargs=3, default=[0,0,0], help="Center of docking box (x,y,z)")
    parser.add_argument("--box_size", type=float, nargs=3, default=[20,20,20], help="Size of docking box (x,y,z)")
    parser.add_argument("--output_dir_docking", type=str, default="output_vina", help="Output directory for docking results")
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
    
    # Clean the FASTA sequence - remove any newlines, spaces, and non-amino acid characters
    # Keep only standard amino acid characters
    cleaned_sequence = ''.join(char.upper() for char in fasta_sequence if char.upper() in 'ACDEFGHIKLMNPQRSTVWY')
    
    log_info(f"Original sequence length: {len(fasta_sequence)}")
    log_info(f"Cleaned sequence length: {len(cleaned_sequence)}")
    log_info(f"Cleaned sequence: {cleaned_sequence[:50]}...{cleaned_sequence[-10:] if len(cleaned_sequence) > 60 else cleaned_sequence[50:]}")
    
    response = requests.post(url, data=cleaned_sequence)
    if response.status_code != 200:
        raise Exception(f"ESMFold API error: {response.status_code} {response.text}")
    pdb_file = os.path.join(output_dir, "receptor.pdb")
    with open(pdb_file, "w") as f:
        f.write(response.text)
    log_info(f"receptor.pdb downloaded and saved to {pdb_file}")
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
        log_info(f"Ligand prepared with Meeko and saved to {output_pdbqt}")
        return True
    except Exception as e:
        log_error(f"Meeko ligand preparation failed: {e}")
        return False


def prepare_ligand_obabel(smiles, output_pdbqt):
    """Alternative ligand preparation using Open Babel with improved PDBQT formatting"""
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
        
        # Convert PDB to SDF first
        sdf_file = output_pdbqt.replace('.pdbqt', '.sdf')
        Chem.MolToMolFile(mol, sdf_file)
        
        # Then convert SDF to PDBQT using obabel with proper options for ligands
        command = ["obabel", sdf_file, "-O", output_pdbqt, "--partialcharge", "gasteiger", "-h"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            log_warning(f"First attempt with SDF failed: {result.stderr}")
            # Try direct PDB to PDBQT conversion
            command = ["obabel", pdb_file, "-O", output_pdbqt, "--partialcharge", "gasteiger", "-h"]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Open Babel error: {result.stderr}")
        
        # Validate the generated PDBQT file
        if not validate_pdbqt_ligand(output_pdbqt):
            raise RuntimeError("Generated PDBQT file is invalid")
        
        # Additional validation: check for Vina compatibility
        if not validate_vina_compatibility(output_pdbqt):
            raise RuntimeError("Generated PDBQT file failed Vina compatibility check")
        
        log_info(f"Ligand prepared with Open Babel and saved to {output_pdbqt}")
        return True
    except Exception as e:
        log_error(f"Open Babel ligand preparation failed: {e}")
        return False

def clean_smiles_for_docking(smiles):
    """Clean and simplify SMILES for better docking compatibility"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Remove stereochemistry if it causes issues
        Chem.rdmolops.RemoveStereochemistry(mol)
        
        # Get largest fragment (remove salts/counterions)
        frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            # Keep largest fragment by atom count
            mol = max(frags, key=lambda x: x.GetNumAtoms())
        
        # Regenerate SMILES
        cleaned_smiles = Chem.MolToSmiles(mol)
        log_info(f"SMILES cleaned: {smiles} -> {cleaned_smiles}")
        return cleaned_smiles
    except Exception as e:
        log_warning(f"SMILES cleaning failed: {e}")
        return smiles

def prepare_ligand_rdkit_manual(smiles, output_pdbqt):
    """Manual PDBQT preparation using RDKit when other methods fail"""
    try:
        # First try to clean the SMILES
        cleaned_smiles = clean_smiles_for_docking(smiles)
        if cleaned_smiles is None:
            raise ValueError("Could not process SMILES")
        
        mol = Chem.MolFromSmiles(cleaned_smiles)
        if mol is None:
            raise ValueError("Could not read cleaned SMILES of ligand")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(mol)
        
        # Generate manual PDBQT content with proper formatting
        pdbqt_content = []
        pdbqt_content.append("REMARK  Generated by RDKit manual PDBQT writer")
        pdbqt_content.append("ROOT")
        
        conf = mol.GetConformer()
        for atom in mol.GetAtoms():
            idx = atom.GetIdx()
            pos = conf.GetAtomPosition(idx)
            element = atom.GetSymbol()
            
            # Proper PDBQT HETATM format for AutoDock Vina
            line = f"HETATM{idx+1:5d}  {element:<4s}LIG A   1    {pos.x:8.3f}{pos.y:8.3f}{pos.z:8.3f}  1.00 20.00           {element}"
            pdbqt_content.append(line)
        
        pdbqt_content.append("ENDROOT")
        
        # Count rotatable bonds
        rotatable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
        pdbqt_content.append(f"TORSDOF {rotatable_bonds}")
        
        with open(output_pdbqt, 'w') as f:
            f.write('\n'.join(pdbqt_content) + '\n')
        
        log_info(f"Ligand prepared with RDKit manual method and saved to {output_pdbqt}")
        return True
    except Exception as e:
        log_error(f"RDKit manual ligand preparation failed: {e}")
        return False

def validate_pdbqt_ligand(pdbqt_file):
    """Validate that the PDBQT file has proper format for ligands"""
    try:
        with open(pdbqt_file, 'r') as f:
            content = f.read()
        
        has_root = "ROOT" in content
        has_endroot = "ENDROOT" in content
        has_atoms = "ATOM" in content or "HETATM" in content
        has_torsdof = "TORSDOF" in content
        
        if not (has_root and has_endroot and has_atoms and has_torsdof):
            log_warning(f"PDBQT validation failed: ROOT={has_root}, ENDROOT={has_endroot}, ATOMS={has_atoms}, TORSDOF={has_torsdof}")
            return False
            
        # Check for problematic patterns
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.strip() == "ROOT" and i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if not next_line.startswith(("ATOM", "HETATM")):
                    log_warning(f"Invalid PDBQT format: ROOT not followed by ATOM/HETATM")
                    return False
        
        log_info(f"PDBQT file validation passed")
        return True
    except Exception as e:
        log_error(f"PDBQT validation failed: {e}")
        return False

def validate_vina_compatibility(pdbqt_file):
    """Additional validation to check Vina compatibility of PDBQT format"""
    try:
        with open(pdbqt_file, 'r') as f:
            lines = f.readlines()
        
        # Check for proper ROOT/ENDROOT structure
        root_found = False
        endroot_found = False
        atoms_between_root = 0
        
        for i, line in enumerate(lines):
            stripped_line = line.strip()
            
            if stripped_line == "ROOT":
                root_found = True
                continue
            elif stripped_line == "ENDROOT":
                endroot_found = True
                break
            elif root_found and (stripped_line.startswith("ATOM") or stripped_line.startswith("HETATM")):
                atoms_between_root += 1
        
        if not (root_found and endroot_found):
            log_warning(f"Vina compatibility check failed: ROOT/ENDROOT structure issue")
            return False
        
        if atoms_between_root == 0:
            log_warning(f"Vina compatibility check failed: No atoms between ROOT/ENDROOT")
            return False
        
        # Check for proper atom line formatting
        for line in lines:
            if line.startswith(("ATOM", "HETATM")):
                # Check if line has proper length and formatting
                if len(line.strip()) < 60:  # Minimum PDBQT line length
                    log_warning(f"Vina compatibility check failed: Short atom line")
                    return False
        
        log_info(f"Vina compatibility check passed: {atoms_between_root} atoms in ROOT section")
        return True
    except Exception as e:
        log_error(f"Vina compatibility check failed: {e}")
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
        log_info(f"Receptor prepared with Meeko and saved to {output_pdbqt}")
        return True
    except Exception as e:
        log_error(f"Meeko receptor preparation failed: {e}")
        return False

def prepare_receptor_obabel(input_pdb, output_pdbqt):
    try:
        cleaned_pdb = input_pdb.replace('.pdb', '_cleaned.pdb')
        clean_pdb_file(input_pdb, cleaned_pdb)
        
        command = ["obabel", cleaned_pdb, "-O", output_pdbqt, "-xr", "--partialcharge", "gasteiger"]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            log_warning(f"First attempt failed: {result.stderr}")
            # Fallback: try without charge calculation
            command = ["obabel", cleaned_pdb, "-O", output_pdbqt, "-xr"]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"Open Babel error: {result.stderr}")
        
        # Validate the generated PDBQT file
        if not validate_pdbqt_receptor(output_pdbqt):
            raise RuntimeError("Generated receptor PDBQT file is invalid")
        
        log_info(f"Receptor prepared with Open Babel and saved to {output_pdbqt}")
        return True
    except Exception as e:
        log_error(f"Open Babel receptor preparation failed: {e}")
        return False

def clean_pdb_file(input_pdb, output_pdb):
    try:
        with open(input_pdb, 'r') as f:
            lines = f.readlines()
        
        cleaned_lines = []
        for line in lines:
            # Keep only essential PDB records
            if line.startswith(('ATOM', 'HETATM', 'CONECT', 'END')):
                # Fix atom names and formatting if needed
                if line.startswith(('ATOM', 'HETATM')):
                    if len(line.strip()) >= 54:  # Minimum length for valid ATOM record
                        cleaned_lines.append(line)
                else:
                    cleaned_lines.append(line)
        
        if cleaned_lines and not cleaned_lines[-1].strip().startswith('END'):
            cleaned_lines.append('END\n')
        
        with open(output_pdb, 'w') as f:
            f.writelines(cleaned_lines)
        
        log_info(f"PDB file cleaned: {len(lines)} -> {len(cleaned_lines)} lines")
        return True
    except Exception as e:
        log_error(f"PDB cleaning failed: {e}")
        return False

def validate_pdbqt_receptor(pdbqt_file):
    try:
        with open(pdbqt_file, 'r') as f:
            content = f.read()
                
        atom_count = content.count("ATOM") + content.count("HETATM")
        
        if atom_count < 10: 
            log_warning(f"Receptor PDBQT has too few atoms: {atom_count}")
            return False
        
        log_info(f"Receptor PDBQT validation passed: {atom_count} atoms")
        return True
    except Exception as e:
        log_error(f"Receptor PDBQT validation failed: {e}")
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
        log_info(f"Detected OS: {os_name}. Downloading Vina executable...")
        for vina_key, vina_exe in vina_paths.items():
            if vina_key == os_name:
                download_url = base_download_url + vina_exe
                response = requests.get(download_url)
                log_info(f"Downloading Vina from {download_url}")
                if response.status_code == 200:
                    with open(vina_path, "wb") as f:
                        f.write(response.content)
                    st = os.stat(vina_path)
                    os.chmod(vina_path, st.st_mode | stat.S_IEXEC)  # Make executable
                    log_success(f"Vina executable downloaded and saved to {vina_path}")
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
    log_info(f"Running Vina: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Vina error: {result.stderr}")
    log_success(f"Docking completed. Results in {out_docked}")

def visualize_docking_results(docked_pdbqt, receptor_pdbqt, output_dir):
    """Visualize docking results using PyMOL or similar tool"""
    try:
        pymol.cmd.reinitialize()  # Clear previous state
        pymol.cmd.load(docked_pdbqt, "docked")
        pymol.cmd.load(receptor_pdbqt, "receptor")
        pymol.cmd.show("sticks", "docked")
        pymol.cmd.show("cartoon", "receptor")
        pymol.cmd.save(os.path.join(output_dir, "docking_results.pse"))
        log_success(f"Docking results visualized and saved to {output_dir}/docking_results.pse")
    except ImportError:
        log_warning("PyMOL not available for visualization. Skipping this step.")

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
                    log_warning(f"Could not parse affinity from line: {line.strip()}")
    if not affinities:
        log_warning("No valid affinities found.")
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
        log_warning("No affinities to plot.")
        return
    bins = np.linspace(min(affinities), max(affinities), 100)
    plt.hist(affinities, bins=bins, color='blue', alpha=0.7)
    plt.xlabel('Binding Affinity (kcal/mol)')
    plt.ylabel('Frequency')
    plt.title('Binding Affinity Distribution')
    plt.grid(axis='y', alpha=0.75, linestyle='--')
    plt.savefig(output_file)
    log_info(f"Affinity histogram saved to {output_file}")

def viability_check(min_affinity, max_affinity, stats):
    """Check if docking results are viable based on affinity statistics"""
    if not stats:
        log_warning("No affinity statistics found.")
        return
    if stats["min"] < min_affinity and stats["max"] > max_affinity:
        log_success("Docking likely viable: good affinity found.")
        return True
    else:
        log_warning("Docking affinities are not very strong, consider reviewing setup or ligand suitability.")
        return False
    

def main():
    args = parse_arguments()
    if os.path.exists(args.output_dir_docking):
        i = 1
        while os.path.exists(f"{args.output_dir_docking}_{i}"):
            i += 1
        args.output_dir_docking = f"{args.output_dir_docking}_{i}"
        print(f"[INFO] Output directory set to {args.output_dir_docking}")
    os.makedirs(args.output_dir_docking, exist_ok=True)

    print("[INFO] Generating receptor PDB from FASTA using ESMFold...")
    pdb_file = fasta_to_pdb_esmfold(args.fasta, args.output_dir_docking)

    print("[INFO] Preparing receptor PDBQT...")
    receptor_pdbqt = os.path.join(args.output_dir_docking, "receptor.pdbqt")

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
    ligand_pdbqt = os.path.join(args.output_dir_docking, "ligand.pdbqt")

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