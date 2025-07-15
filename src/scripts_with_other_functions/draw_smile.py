from rdkit import Chem
from rdkit.Chem import Draw

def save_molecule_image(smile, filename):
    """
    Convert a SMILES string to a molecule image and save it to a file.
    
    Parameters:
        smile (str): The SMILES string representing the molecule.
        filename (str): The path and name of the file where the image will be saved. Defaults to 'molecule.png'.
    """
    molecule = Chem.MolFromSmiles(smile)
    
    if molecule is not None:
        img = Draw.MolToImage(molecule, size=(400, 300))
        img.save(filename)
    else:
        print(f"[WARNING] Invalid SMILES string: {smile}. Unable to create molecule image.")
        raise ValueError("Invalid SMILES string provided.")
# Example usage
smile = "c1C[C@H](NC(=O)[C@@H]1C[C@@H](O)CN1C(=O)[C@@H](NC(=O)CCCCCCCCCNC(=O)CCc1ccc(-c2ccc(/C=C/c3ccccc3)c3ccccc32)on1)C(C)C)C(=O)NCC(N)=O"
save_molecule_image(smile, "molecule.png")
