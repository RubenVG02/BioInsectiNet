from rdkit import Chem
import random
from rdkit.Chem.rdchem import RWMol
import numpy as np

parents=["O=C(c1ccccc1)c1ccccc1C","Oc1cc(O)c(OC(=O)Nc2ccccc2)cc1"]

def mutations(smile="", mutation_rate=0.1):

    '''
    Function to mutate a molecule in order to obtain a new molecule with a better affinity to the target.

    Parameters:
    -smiles: Sequence of the molecule in smile format obtained from the childs function.
    -mutation_rate: Probability of mutation of an atom in the molecule.

    '''
    atoms=[6, 14, 5, 7, 15, 8, 16, 9, 17, 35, 53]

    aromatic_atoms=[6,7,15,8,16]

    len_smile=Chem.MolFromSmiles(smile).GetNumAtoms()

    copy=smile  #We copy the molecule in order to not modify the original molecule

    mol1=Chem.MolFromSmiles(smile)
    Chem.Kekulize(mol1)
    mol1=RWMol(mol1)
    
    child_generated=[]
    while len(child_generated)<5:
        for molecule in copy:
            if random.uniform(0,1) <= mutation_rate:
                atom_to_remove=np.random.randint(0, len_smile) #random atom to remove from the molecule
                try:
                    valence_atom=mol1.GetAtomWithIdx(atom_to_remove).GetTotalValence() #valence of the atom to remove
                except:
                    pass
                print(valence_atom)
                if mol1.GetAtomWithIdx(atom_to_remove).GetIsAromatic():
                    if valence_atom==2:
                        mol1.ReplaceAtom(atom_to_remove, Chem.Atom(aromatic_atoms[np.random.randint(1, 5)]))
                    elif valence_atom==3:
                        mol1.ReplaceAtom(atom_to_remove, Chem.Atom(aromatic_atoms[np.random.randint(0, 3)]))
                    else:
                        continue
                    mol1.GetAtomWithIdx(atom_to_remove).SetIsAromatic(True)
                else:
                    if valence_atom==1:
                        mol1.ReplaceAtom(atom_to_remove, Chem.Atom(atoms[np.random.randint(0, 11)])) 
                    elif valence_atom==2:
                        mol1.ReplaceAtom(atom_to_remove, Chem.Atom(atoms[np.random.randint(0, 7)]))  
                    elif valence_atom==3:
                        mol1.ReplaceAtom(atom_to_remove, Chem.Atom(atoms[np.random.randint(0, 5)]))
                    elif valence_atom==4:
                        mol1.ReplaceAtom(atom_to_remove, Chem.Atom(atoms[np.random.randint(0, 2)]))  
            if Chem.MolToSmiles(mol1):
                if Chem.MolToSmiles(mol1) not in child_generated:
                    child_generated.append(Chem.MolToSmiles(mol1))

               
    print(child_generated) 

for i in parents:
    mutations(i)

