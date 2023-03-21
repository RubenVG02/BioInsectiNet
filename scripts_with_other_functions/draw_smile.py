from rdkit import Chem
from rdkit.Chem import Draw


smile = "B=C(CCC)Cp1cnnc1CBC(=B)c1ccc(I)cc1N"

FILENAME="moleule5.png"

molecule = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecule, filename=FILENAME,
                    size=(400, 300))


