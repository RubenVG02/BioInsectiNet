from rdkit import Chem
from rdkit.Chem import Draw


smile = "Cc1ccc(-c2ccc(C)c([SiH2]C(=[SiH2])CC([SiH3])C(=O)NCc3ccccc3)n2)cc1"

FILENAME="moleule.png"

molecule = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecule, filename=FILENAME,
                    size=(400, 300))


