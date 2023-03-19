from rdkit import Chem
from rdkit.Chem import Draw


smile = "C=C(F)c1ccc(C(=N)Cc2ncc(C(=O)NC(C)C(C)CC)[pH]2)cc1"

FILENAME="moleule2.png"

molecule = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecule, filename=FILENAME,
                    size=(400, 300))


