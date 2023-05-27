
'''
This script is used to convert a 3D molecule from SDF format to STL format.

In order to run this script, you need to install PyMOl first.

You can change the output format by changing the extension of the output file in cmd.save().
'''

import __main__
__main__.pymol_argv = [ 'pymol', '-qc'] # Quiet and no GUI

import pymol

pymol.finish_launching()

from pymol import cmd

cmd.load('molecule_sdf.sdf') 
cmd.save('molecule_stl.stl')
cmd.quit()

