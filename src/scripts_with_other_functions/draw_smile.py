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
        print("Error: Invalid SMILES string or molecule could not be created.")

# Example usage
smile = "Nc1ccnc2c1ccn2[C@@H]1O[C@H](CO)C[C@H]1O"
save_molecule_image(smile, "molecule.png")
