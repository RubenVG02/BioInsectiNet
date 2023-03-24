from rdkit import Chem
from rdkit.Chem import Draw


smile = "N=C(Bc1pc2ccccc2s1)c1ccccc1CC(=O)B1CCCCC1"

FILENAME="moleule5.png"

molecule = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecule, filename=FILENAME,
                    size=(400, 300))


