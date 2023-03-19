from rdkit import Chem 
from rdkit.Chem import Draw
from rdkit.Chem.rdchem import RWMol
from rdkit.Chem import Descriptors
from check_affinity import calculate_affinity
import random   
import numpy as np
import csv
from affinity_with_target_and_generator import find_candidates
import time


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
                    score=[]
                    for i in initial_population:
                        i=i.replace("@", "").replace("/", "")
                        value=calculate_affinity(smile=i, fasta=target)
                        score.append(value)

    if ".txt" in initial_population:
        with open(initial_population, "r") as file:
            score=[]
            for row in file:
                value=calculate_affinity(smile=row, fasta=target)
                score.append(value)
    total=zip(initial_population, score)
    total=sorted(total, key=lambda x: x[1])
    parents=total[:bests]
    return parents            
  

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
            
def childs(parents):
    '''
    Function to cross two smile sequences in order to obtain two new molecules. Function used when ic50 value does not improve during the generations.
    
    '''
    crossover_point=random.randint(0, len(parents[0])-1)
    child1 = parents[0][:crossover_point] + parents[1][crossover_point:]
    child2 = parents[1][:crossover_point] + parents[0][crossover_point:]
    print(child1, child2)
    child1= Chem.MolFromSmiles(child1)
    child2= Chem.MolFromSmiles(child2)
    if child1 is not None and child2 is not None:
        return child1, child2
    else:
        childs() 

def mutations(smile="", mutation_rate=0.1):

    '''
    Function to mutate a molecule in order to obtain a new molecule with a better affinity to the target.

    Parameters:
    -smiles: Sequence of the molecule in smile format obtained from the parents generation.
    -mutation_rate: Probability of mutation of an atom in the molecule.

    '''
    atoms=[6, 5, 7, 15, 8, 16, 9, 17, 35, 53]

    aromatic_atoms=[6, 7, 15, 8, 16]

    len_smile=Chem.MolFromSmiles(smile).GetNumAtoms()

    copy=smile  #We copy the molecule in order to not modify the original molecule

    mol1=Chem.MolFromSmiles(copy)
    Chem.SanitizeMol(mol1)
    Chem.Kekulize(mol1)
    mol1=RWMol(mol1)

    
    
    child_generated=[]
    while len(child_generated)<5:
        for molecule in copy:
            if random.uniform(0,1) <= mutation_rate:
                atom_to_remove=np.random.randint(0, len_smile) #random atom to remove from the molecule
                try:
                    valence_atom=mol1.GetAtomWithIdx(atom_to_remove).GetTotalValence() #valence of the atom to remove
                    print(valence_atom)
                except:
                    pass
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
                        mol1.ReplaceAtom(atom_to_remove, Chem.Atom(atoms[np.random.randint(0, 10)])) 
                    elif valence_atom==2:
                        mol1.ReplaceAtom(atom_to_remove, Chem.Atom(atoms[np.random.randint(0, 6)]))  
                    elif valence_atom==3:
                        mol1.ReplaceAtom(atom_to_remove, Chem.Atom(atoms[np.random.randint(0, 4)]))
                    elif valence_atom==4:
                        mol1.ReplaceAtom(atom_to_remove, Chem.Atom(atoms[np.random.randint(0, 1)]))  
            try:
                Chem.SanitizeMol(mol1)
                Chem.Kekulize(mol1)
            except Chem.rdchem.KekulizeException:
                pass
            if Chem.MolToSmiles(mol1):
                if Chem.MolToSmiles(mol1) not in child_generated:
                    if check_druglikeness(Chem.MolToSmiles(mol1)):
                        
                        child_generated.append(Chem.MolToSmiles(mol1))        
    return child_generated
                
def file_preparation(file_path="", name_file="", headers=[]):
    '''
    Function to create the .csv file to which the obtained data will be added
    
    Parameters:
    -headers: Names of the columns we want to use
    '''
    with open(f"{name_file}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)     


def genetic_algorithm(target="", initial_pop_path=r"", objective_ic50=20, generations=100, bests=2, path_save=r"", save_since=20, name_file=""):
    '''
    Function to find the best molecule to bind to a target protein using a genetic algorithm.

    Parameters:
    -target: Sequence of the target protein in fasta format.
    -initial_pop_path: Path of the initial population of smile molecules. If this file does not exist, the function will create it.
    -objective_ic50: Value of the affinity of the target protein that we want to obtain. By default, it is 20.
    -generations: Number of generations of the genetic algorithm. By default, it is 100.
    -bests: Number of best molecules that we want to select from each generation. By default, it is 5.
    -path_save: Path where we want to save the best molecule obtained in each generation. By default, it is the current directory.
    -save_since: Since which ic50 value we want to save the best molecule obtained in each generation. By default, it is 20.
    -name_file: Name of the file where we want to save the best molecule obtained depending on best and save_since. By default, it is "resultados.csv".
    '''
    parents=select_parents(initial_population=initial_pop_path, target=target, bests=bests)  #We select the best molecules from the initial population
    file_preparation(file_path=path_save, name_file=name_file, headers=["SMILE", "Affinity"])

    all_bests=[] #We create a list to save the best molecules obtained in each generation in order to compare if we need to use the childs function
    sum_not_improve=0 #We create a variable to count the number of generations in which the best molecule has not improved
    for gen in range(generations):
        new_generation=[]
        total=[]
        if sum_not_improve >= 2: #If the best molecule has not improved for 10 generations, we use the childs function
            print("Using childs function")
            parents= childs(parents=parents[:2], target=target, mutation_rate=0.1)
            sum_not_improve=0
        else:
            parents=[i[0] for i in parents]
        for i in parents:
            mutation=mutations(smile=i, mutation_rate=0.1)
            new_generation.extend(mutation)

        score=[]
        print(new_generation)
        
        for smile in new_generation:
            smile=str(smile).replace("@", "").replace("\\","").replace("/", "").replace(".", "")
            value=calculate_affinity(smile=smile, fasta=target)
            score.append(value)
        total=zip(new_generation, score)
        total=sorted(total, key=lambda x: x[1])
        with open(path_save, "a") as file:
            for i in total:
                if i[1] <= save_since:
                    file.write(f"{i[0]}, {i[1]}\n")
        if compare_ic50(list_score=total, objective_ic50=objective_ic50) is not False:
            best_individual, affinity= compare_ic50(list_score=total, objective_ic50=objective_ic50)
            print("Generation:", gen+1)
            print("Best SMILE sequence obtained:", best_individual)
            print("Fitness:", affinity)
            print("--------")
            break
        else:
            parents=total[:bests]
            if parents[0][1] not in all_bests:
                all_bests.append(parents[0][0], parents[0][1])
            if parents[0][1]< min(all_bests[1]):
                sum_not_improve+=1
            print("\n\n\n" + "Generation:", gen+1)
            print("Best SMILE sequence obtained:", parents[0][0])
            print("Fitness:", parents[0][1])
            print("\n\n\n")
            time.sleep(4)
            continue
    


def compare_ic50(list_score, objective_ic50):

    '''
    Function to compare the affinity of the molecules with the objective affinity.

    Parameters:
    -list_score: List of the affinity of the molecules.
    -objective_ic50: Value of the affinity of the target protein that we want to obtain. 
    '''
    for i in list_score:
        if i[1] <= objective_ic50:
            return i[0], i[1]
        else:
            return False
    

genetic_algorithm(target="MSFVHLQVHSGYSLLNSAAAVEELVSEADRLGYASLALTDDHVMYGAIQFYKACKARGINPIIGLTASVFTDDSELEAYPLVLLAKSNTGYQNLLKISSVLQSKSKGGLKPKWLHSYREGIIAITPGEKGYIETLLEGGLFEQAAQASLEFQSIFGKGAFYFSYQPFKGNQVLSEQILKLSEETGIPVTATGDVHYIRKEDKAAYRCLKAIKAGEKLTDAPAEDLPDLDLKPLEEMQNIYREHPEALQASVEIAEQCRVDVSLGQTRLPSFPTPDGTSADDYLTDICMEGLRSRFGKPDERYLRRLQYELDVIKRMKFSDYFLIVWDFMKHAHEKGIVTGPGRGSAAGSLVAYVLYITDVDPIKHHLLFERFLNPERVSMPDIDIDFPDTRRDEVIQYVQQKYGAMHVAQIITFGTLAAKAALRDVGRVFGVSPKEADQLAKLIPSRPGMTLDEARQQSPQLDKRLRESSLLQQVYSIARKIEGLPRHASTHAAGVVLSEEPLTDVVPLQEGHEGIYLTQYAMDHLEDLGLLKMDFLGLRNLTLIESITSMIEKEENIKIDLSSISYSDDKTFSLLSKGDTTGIFQLESAGMRSVLKRLKPSGLEDIVAVNALYRPGPMENIPLFIDRKHGRAPVHYPHEDLRSILEDTYGVIVYQEQIMMIASRMAGFSLGEADLLRRAVSKKKKEILDRERSHFVEGCLKKEYSVDTANEVYDLIVKFANYGFNRSHAVAYSMIGCQLAYLKAHYPLYFMCGLLTSVIGNEDKISQYLYEAKGSGIRILPPSVNKSSFPFTVENGSVRYSLRAIKSVGVSAVKDIYKARKEKPFEDLFDFCFRVPSKSVNRKMLEALIFSGAMDEFGQNRATLLASIDVALEHAELFAADDDQMGLFLDESFSIKPKYVETEELPLVDLLAFEKETLGIYFSNHPLSAFRKQLTAQGAVSILQAQRAVKRQLSLGVLLSKIKTIRTKTGQNMAFLTLSDETGEMEAVVFPEQFRQLSPVLREGALLFTAGKCEVRQDKIQFIMSRAELLEDMDAEKAPSVYIKIESSQHSQEILAKIKRILLEHKGETGVYLYYERQKQTIKLPESFHINADHQVLYRLKELLGQKNVVLKQW", initial_pop_path=r"Topoisomerasa_4(Aeruginosa).csv", objective_ic50=20, generations=100, bests=2, path_save=r"resultados.csv", save_since=40, name_file="resultados")
