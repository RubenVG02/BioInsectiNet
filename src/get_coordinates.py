from rdkit import Chem
from rdkit.Chem import AllChem

def generate_sdf_from_smiles(smile, name):
    '''
    Generate an SDF file from a SMILES string using RDKit.
    
    Parameters:
        smile (str): SMILES sequence of the molecule.
        name (str): Name of the output SDF file (without extension).
    
    Returns:
        str: Message indicating the result of the file creation.
    '''
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            print("Invalid SMILES")
            return "Invalid SMILES"
        
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol) 
        
        writer = Chem.SDWriter(f"{name}.sdf") 
        writer.write(mol)
        writer.close()
        
        return "SDF file created"
    
    except Exception as e:
        print(f"Error: {e}")
        return f"Error: {e}"

if __name__ == "__main__":
    smile = "Cc1ccc2[nH]c(-c3ccc3c(c4)OCO4)nc2c1"  
    result = generate_sdf_from_smiles(smile, "example_molecule")
    print(result)