def compare_smiles(file1, file2):
    """
    Function to compare the number of unique SMILESS strings in two databases.
    
    Parameters:
        file1 (path): Path to the first file containing SMILES strings.
        file2 (path): Path to the second file containing SMILES strings (optional).
        
    Returns:
        List of SMILES strings that are in the first file but not in the second file.
    """
    
    smiles1 = open(file1).read().splitlines()
    smiles2 = open(file2).read().splitlines() if file2 else []
    
    difference = list(set(smiles1) - set(smiles2))
    
    return difference


