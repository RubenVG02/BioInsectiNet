from rdkit import Chem
import random
from rdkit.Chem.rdchem import RWMol

parents=["O=C(c1ccccc1)c1ccccc1C","Oc1cc(O)c(OC(=O)Nc2ccccc2)cc1"]

print(Chem.MolFromSmiles(parents[0]).GetNumAtoms())
print(len(parents[0]))
RWMol(Chem.MolFromSmiles(parents[0]))
def mutations(smiles=parents, mutation_rate=0.1):

    '''
    Function to mutate a molecule in order to obtain a new molecule with a better affinity to the target.

    Parameters:
    -smiles: Sequence of the molecule in smile format obtained from the childs function.
    -mutation_rate: Probability of mutation of an atom in the molecule.

    '''
    atoms=[6, 14, 5, 7, 15, 8, 16, 9, 17, 35, 53]

    aromatic_atoms=[6,7,15,8,16]

    len_smile1=Chem.MolFromSmiles(smiles[0]).GetNumAtoms()
    len_smile2=Chem.MolFromSmiles(smiles[1]).GetNumAtoms()
    print(len_smile1)
    copy=smiles.copy()

    mol1=Chem.MolFromSmiles(smiles[0])
    mol1=RWMol(mol1)
    
    for molecule in copy:
        for atom in molecule:
            if random.uniform(0,1) <= mutation_rate:
                atom_to_remove=random.randint(0, len_smile1) #random atom to remove from the molecule
                valence_atom=mol1.GetAtomWithIdx(atom_to_remove).GetTotalValence()
                print(valence_atom)
                if mol1.GetAtomWithIdx(atom_to_remove).GetIsAromatic():
                    if valence_atom==2:
                        mol1.ReplaceAtom(atom_to_remove, random.choice(aromatic_atoms[random.randint(1, 5)]))
                    elif valence_atom==3:
                        mol1.ReplaceAtom(atom_to_remove, random.choice(aromatic_atoms[random.randint(0, 3)]))
                else:
                    if valence_atom==1:
                        mol1.ReplaceAtom(atom_to_remove, random.choice(atoms[random.randint(0, 11)])) 
                    elif valence_atom==2:
                        mol1.ReplaceAtom(atom_to_remove, random.choice(atoms[random.randint(0, 7)]))  
                    elif valence_atom==3:
                        mol1.ReplaceAtom(atom_to_remove, random.choice(atoms[random.randint(0, 5)]))
                    '''elif valence_atom==4:
                        mol1.ReplaceAtom(atom_to_remove, random.choice(atoms[random.randint(0, 2)]))   '''  
    
    print(Chem.MolToSmiles(mol1))
mutations()

