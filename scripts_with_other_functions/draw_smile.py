from rdkit import Chem
from rdkit.Chem import Draw


smile = "CCC(C)[C@@H](C)NC(=O)c1ccc(NC(=O)c2ccc(C(=N)F)cc2)[nH]1"

FILENAME="moleule4.png"

molecule = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecule, filename=FILENAME,
                    size=(400, 300))


