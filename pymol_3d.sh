
cd ${PWD} # current working directory. If not working, use the absolute path
open -a PyMOL
cmd.load('molecule_sdf.sdf')
cmd.save('molecule_pdb.stl')

#load Users/rubenvg/Desktop/antiinsecticides/Fungic_insecticides/molecule_sdf.sdf
