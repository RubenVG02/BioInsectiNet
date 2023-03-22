from rdkit import Chem
from rdkit.Chem import Draw


smile = "C=C(CCC)CC1C=CC=C1CNC(=O)c1ccc(P)cc1S"

FILENAME="moleule5.png"

molecule = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecule, filename=FILENAME,
                    size=(400, 300))


