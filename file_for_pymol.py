## This script will transform our sdf file into a stl file using pymol
## We will use this script using the pymol terminal, and the stl file will be saved in the same folder as the sdf file

cmd.load('molecule_sdf.sdf')
cmd.save('molecule_stl.stl')
cmd.quit()