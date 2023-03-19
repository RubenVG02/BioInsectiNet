from rdkit import Chem
import random
from rdkit.Chem.rdchem import RWMol
import numpy as np
from rdkit.Chem import Descriptors
from check_affinity import calculate_affinity
import csv
from affinity_with_target_and_generator import find_candidates

parents=["O=C(c1ccccc1)c1ccccc1C","Oc1cc(O)c(OC(=O)Nc2ccccc2)cc1"]

def check_druglikeness(smile=""):
    '''
    Function to check if a molecule is druglike or not. If it is not druglike, it will be removed from the population.

    Parameters:
    -smile: Sequence of the molecule in smile format.

    '''
    mol1=Chem.MolFromSmiles(smile)
    if mol1 is not None:
        if Descriptors.ExactMolWt(mol1) < 500 and Descriptors.MolLogP(mol1) < 5 and Descriptors.NumHDonors(mol1) < 5 and Descriptors.NumHAcceptors(mol1) < 10:
                        #All the conditions that a molecule must meet to be considered drug-like  
            return True

def mutations(smile="", mutation_rate=0.1):

    '''
    Function to mutate a molecule in order to obtain a new molecule with a better affinity to the target.

    Parameters:
    -smiles: Sequence of the molecule in smile format obtained from the childs function.
    -mutation_rate: Probability of mutation of an atom in the molecule.

    '''
    atoms=[6, 14, 5, 7, 15, 8, 16, 9, 17, 35, 53]

    aromatic_atoms=[6,7,15,8,16]

    len_smile=Chem.MolFromSmiles(smile).GetNumAtoms()

    copy=smile  #We copy the molecule in order to not modify the original molecule

    mol1=Chem.MolFromSmiles(smile)
    Chem.Kekulize(mol1)
    mol1=RWMol(mol1)
    
    child_generated=[]
    while len(child_generated)<5:
        for molecule in copy:
            if random.uniform(0,1) <= mutation_rate:
                atom_to_remove=np.random.randint(0, len_smile) #random atom to remove from the molecule
                try:
                    valence_atom=mol1.GetAtomWithIdx(atom_to_remove).GetTotalValence() #valence of the atom to remove
                except:
                    pass
                print(valence_atom)
                if mol1.GetAtomWithIdx(atom_to_remove).GetIsAromatic():
                    if valence_atom==2:
                        mol1.ReplaceAtom(atom_to_remove, Chem.Atom(aromatic_atoms[np.random.randint(1, 5)]))
                    elif valence_atom==3:
                        mol1.ReplaceAtom(atom_to_remove, Chem.Atom(aromatic_atoms[np.random.randint(0, 3)]))
                    else:
                        continue
                    mol1.GetAtomWithIdx(atom_to_remove).SetIsAromatic(True)
                else:
                    if valence_atom==1:
                        mol1.ReplaceAtom(atom_to_remove, Chem.Atom(atoms[np.random.randint(0, 11)])) 
                    elif valence_atom==2:
                        mol1.ReplaceAtom(atom_to_remove, Chem.Atom(atoms[np.random.randint(0, 7)]))  
                    elif valence_atom==3:
                        mol1.ReplaceAtom(atom_to_remove, Chem.Atom(atoms[np.random.randint(0, 5)]))
                    elif valence_atom==4:
                        mol1.ReplaceAtom(atom_to_remove, Chem.Atom(atoms[np.random.randint(0, 2)]))  
            if Chem.MolToSmiles(mol1):
                if Chem.MolToSmiles(mol1) not in child_generated:
                    if check_druglikeness(Chem.MolToSmiles(mol1)):
                        child_generated.append(Chem.MolToSmiles(mol1))

               
    print(child_generated) 

def select_parents(initial_population=r"/Users/rubenvg/Desktop/antiinsecticides/Fungic_insecticides/Topoisomerasa_4(Aeruginosa).csv", target="", bests=2):

    '''
    Function to select the parents of the first generations of the genetic algorithm. The parents are the molecules with the best affinity to the target.

    Parameters:
    -initial_population: Path of the file with the initial population of molecules. If it is not specified, the function will use the function find_candidates to obtain the initial population.
    -target: Sequence of the target in FASTA format.
    
    '''
    if not initial_population:
        initial_population=find_candidates(return_path=True,max_molecules=10,db_smiles=False,target=target,draw_minor=False,generate_qr=False,upload_to_mega=False)
    if not ".txt" in initial_population:
        with open(initial_population, "r") as file:
                    reader = csv.reader(file)
                    initial_population = [row[0] for row in reader][1:]
                    print(initial_population)
                    score=[]
                    for i in initial_population:
                        i=i.replace("@", "").replace("/", "")
                        value=calculate_affinity(target, i)
                        score.append(value)

    if ".txt" in initial_population:
        with open(initial_population, "r") as file:
            score=[]
            for row in file:
                value=calculate_affinity(target, row)
                score.append(value)
    total=zip(initial_population, score)
    total=sorted(total, key=lambda x: x[1])
    parents=total[:bests]
    return parents            
  
'''target="MSFVHLQVHSGYSLLNSAAAVEELVSEADRLGYASLALTDDHVMYGAIQFYKACKARGINPIIGLTASVFTDDSELEAYPLVLLAKSNTGYQNLLKISSVLQSKSKGGLKPKWLHSYREGIIAITPGEKGYIETLLEGGLFEQAAQASLEFQSIFGKGAFYFSYQPFKGNQVLSEQILKLSEETGIPVTATGDVHYIRKEDKAAYRCLKAIKAGEKLTDAPAEDLPDLDLKPLEEMQNIYREHPEALQASVEIAEQCRVDVSLGQTRLPSFPTPDGTSADDYLTDICMEGLRSRFGKPDERYLRRLQYELDVIKRMKFSDYFLIVWDFMKHAHEKGIVTGPGRGSAAGSLVAYVLYITDVDPIKHHLLFERFLNPERVSMPDIDIDFPDTRRDEVIQYVQQKYGAMHVAQIITFGTLAAKAALRDVGRVFGVSPKEADQLAKLIPSRPGMTLDEARQQSPQLDKRLRESSLLQQVYSIARKIEGLPRHASTHAAGVVLSEEPLTDVVPLQEGHEGIYLTQYAMDHLEDLGLLKMDFLGLRNLTLIESITSMIEKEENIKIDLSSISYSDDKTFSLLSKGDTTGIFQLESAGMRSVLKRLKPSGLEDIVAVNALYRPGPMENIPLFIDRKHGRAPVHYPHEDLRSILEDTYGVIVYQEQIMMIASRMAGFSLGEADLLRRAVSKKKKEILDRERSHFVEGCLKKEYSVDTANEVYDLIVKFANYGFNRSHAVAYSMIGCQLAYLKAHYPLYFMCGLLTSVIGNEDKISQYLYEAKGSGIRILPPSVNKSSFPFTVENGSVRYSLRAIKSVGVSAVKDIYKARKEKPFEDLFDFCFRVPSKSVNRKMLEALIFSGAMDEFGQNRATLLASIDVALEHAELFAADDDQMGLFLDESFSIKPKYVETEELPLVDLLAFEKETLGIYFSNHPLSAFRKQLTAQGAVSILQAQRAVKRQLSLGVLLSKIKTIRTKTGQNMAFLTLSDETGEMEAVVFPEQFRQLSPVLREGALLFTAGKCEVRQDKIQFIMSRAELLEDMDAEKAPSVYIKIESSQHSQEILAKIKRILLEHKGETGVYLYYERQKQTIKLPESFHINADHQVLYRLKELLGQKNVVLKQW"
parents=select_parents(target=target)
parents=[i[0] for i in parents]
print(parents)'''

len_smile=Chem.MolFromSmiles("Cc1ccc(-c2ccc(C)c(NC(=O)C[C@@H](C)C(=O)NCc3ccccc3)n2)cc1").GetNumAtoms()
print(len_smile)
