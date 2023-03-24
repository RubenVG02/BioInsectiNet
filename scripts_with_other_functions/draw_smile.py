from rdkit import Chem
from rdkit.Chem import Draw


smile = "C=C(Cp1nnnc1[SiH2]OC(=O)c1ccc(P)cc1Cl)O[SiH2][SiH3]"

FILENAME="moleule5.png"

molecule = Chem.MolFromSmiles(smile)
Draw.MolToImageFile(molecule, filename=FILENAME,
                    size=(400, 300))


