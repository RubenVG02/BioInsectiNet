from rdkit import Chem
from rdkit.Chem import Draw


smile = "C=C(Nc1ccccc1)[SiH2]c1c(N)cc(O)cc1"

FILENAME="moleule.png"

molecule = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecule, filename=FILENAME,
                    size=(400, 300))


