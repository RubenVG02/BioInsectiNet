from rdkit import Chem
from rdkit.Chem import Draw


smile = "OC(=O)[C@@H](C)[NH2+][C@@H](Cc1c[nH]c2ccccc12)C(=O)NCC(C)C"

FILENAME=""

molecule = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecule, filename=FILENAME,
                    size=(400, 300))


