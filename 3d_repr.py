import requests

def obtain_sdf(smile="", name="molecule_sdf"):
    
    '''
    Function to obtain sdf file of an smile sequence in order to use it on PyMol
    
    Parameters:
    -smile: 
    
    
    '''
    request=requests.get(f"https://cactus.nci.nih.gov/chemical/structure/{smile}/file?format=sdf&get3d=True")
    sdf_file=request.text
    print(sdf_file)
    if "Page not found" in sdf_file:
        print("SDF file not found")
        return "SDF file not found"
    else:
        with open(f"{name}.sdf","w") as f:
            f.write(sdf_file)
        print("SDF file created")
    
obtain_sdf(smile="C=CCNc1nnc(SCc2ccccc2C)cc1OC(=O)CS")
