from rdkit import Chem
from rdkit.Chem import Draw


smile = "CC[C@@H](C)C(C)NC(=O)c1ccc(NC(=O)c2ccc(C(C)=O)cc2)[pH]1"

FILENAME="moleule5.png"

molecule = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecule, filename=FILENAME,
                    size=(400, 300))


