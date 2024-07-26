import requests

def obtain_sdf(smile="", name="molecule_sdf"):
    """
    Obtain the SDF file of a molecule given its SMILES sequence.

    Parameters:
        smile (str): SMILES sequence of the molecule.
        name (str): Name of the output SDF file (without extension).

    Returns:
        str: Message indicating the result of the file creation.
    """

    url = f"https://cactus.nci.nih.gov/chemical/structure/{smile}/file?format=sdf&get3d=True"
    
    response = requests.get(url)
    
    sdf_file = response.text
    
    if "Page not found" in sdf_file:
        print("SDF file not found")
        return "SDF file not found"
    else:
        with open(f"{name}.sdf", "w") as f:
            f.write(sdf_file)
        print("SDF file created")
        return "SDF file created"
