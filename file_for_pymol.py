## This script will transform our sdf file into a stl file using pymol
## We will use the following command: pymol -c file_for_pymol.py

cmd.load('molecule_sdf.sdf')
cmd.save('molecule_stl.stl')
cmd.quit()