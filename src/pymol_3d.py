"""
This script is used to convert a 3D molecule from SDF format to STL format.

In order to run this script, you need to install PyMOL first. It can't be run directly through the terminal, but needs to be run through PyMOL.

You can change the output format by changing the extension of the output file in cmd.save().
"""

import __main__
import os
import pymol
import requests
from pymol import cmd

# Configure PyMOL to run in quiet mode with no GUI
__main__.pymol_argv = ['pymol', '-qc']
pymol.finish_launching()

def obtain_sdf(smile="", name="molecule_sdf"):
    """
    Function to obtain an SDF file of a molecule given its SMILES sequence.

    Parameters:
        smile (str): SMILES sequence of the molecule.
        name (str): Name of the SDF file to be saved (without extension).
    
    """
    response = requests.get(f"https://cactus.nci.nih.gov/chemical/structure/{smile}/file?format=sdf&get3d=True")
    sdf_file = response.text
    
    if "Page not found" in sdf_file:
        print("SDF file not found")
        return "SDF file not found"
    else:
        with open(f"{name}.sdf", "w") as f:
            f.write(sdf_file)
        print("SDF file created")
        return "SDF file created"

def get_pymol(smile="", name="molecule_sdf"):

    """
    Convert an SDF file to STL format using PyMOL.

    Parameters:
        smile (str): SMILES sequence of the molecule.
        name (str): Name of the SDF file to be used (without extension).
    """

    if not os.path.isfile(f"{name}.sdf"):
        obtain_sdf(smile=smile, name=name)
    
    cmd.load(f"{name}.sdf")
    
    cmd.save('molecule_stl.stl')
    
    cmd.quit()

# Example usage
get_pymol(smile='CC(=O)OC1=CC=CC=C1C(=O)O', name='molecule_sdf')
