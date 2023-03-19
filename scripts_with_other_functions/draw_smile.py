from rdkit import Chem
from rdkit.Chem import Draw


smile = "C=C([O-])c1ccc(C(=O)Bc2ncc(C(=O)N[C@H](C)[C@H](C)CC)o2)cc1"

FILENAME="moleule.png"

molecule = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecule, filename=FILENAME,
                    size=(400, 300))


