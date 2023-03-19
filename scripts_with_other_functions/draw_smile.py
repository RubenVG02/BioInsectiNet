from rdkit import Chem
from rdkit.Chem import Draw


smile = "CC[C@@H](C)C(C)NC(=O)c1cpc(NC(=O)c2ccc(C(=O)S)cc2)s1"

FILENAME="moleule3.png"

molecule = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecule, filename=FILENAME,
                    size=(400, 300))


