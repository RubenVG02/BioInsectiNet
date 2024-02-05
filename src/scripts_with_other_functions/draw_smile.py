from rdkit import Chem
from rdkit.Chem import Draw


smile = "Nc1ccnc2c1ccn2[C@@H]1O[C@H](CO)C[C@H]1O"

FILENAME=""

molecule = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecule, filename=FILENAME,
                    size=(400, 300))


