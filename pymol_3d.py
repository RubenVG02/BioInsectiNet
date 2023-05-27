
'''
This script is used to convert a 3D molecule from SDF format to STL format.

In order to run this script, you need to install PyMOl first.

You can change the output format by changing the extension of the output file in cmd.save().
'''


import __main__
import os
__main__.pymol_argv = [ 'pymol', '-qc'] # Quiet and no GUI
import pymol
import requests
pymol.finish_launching()
from pymol import cmd



def obtain_sdf(smile="", name="molecule_sdf"):
    """
    Function to obtain sdf file of an smile sequence in order to use it on PyMol
    
    Parameters:
    -smile: Sequence of the molecule in smile format.
    -name: Name of the sdf file.
    """
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
    

def get_pymol(smile="", name="molecule_sdf"):
    if not name in os.listdir():
        obtain_sdf(smile=smile, name=name) # Change the smile and output name here
    cmd.load(f"{name}.sdf")
    cmd.save('molecule_stl.stl')
    cmd.quit()


get_pymol(smile='CC(=O)OC1=CC=CC=C1C(=O)O', name='molecule_sdf') # Change the smile and output name here