from rdkit import Chem
from rdkit.Chem import Draw


smile = "C[SiH](C[Si](=O)c1cnc(CC(=O)c2ccc(C(=O)[O-])cc2)s1)[C@H]([SiH3])C[SiH3]"

FILENAME="moleule.png"

molecule = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecule, filename=FILENAME,
                    size=(400, 300))


