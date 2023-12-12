
def compare(data1=r"", data2=r""):
    '''
  Function to compare the amount of elements that two SMILES db have, so you can use them for a given model
    
    Parameters:
        -data1: Path of the first file with Smiles
        -data2: Path of the second file with Smiles
        
    Returns:
        -list with both files
    '''


    data1 = open(data1).read()
    data2 = open(data2).read()
    
    diferència=list(set(data1)-set(data2))
    return diferència
compare()