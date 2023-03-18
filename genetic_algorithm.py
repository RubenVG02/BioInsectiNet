from rdkit import Chem 
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from check_affinity import calculate_affinity
import random   
import csv
from affinity_with_target_and_generator import find_candidates


def select_parents(initial_population=r"", target=""):

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
                    initial_population = [row for row in reader[1]] #mirar si es 1 o 0
                    score=[]
                    for i in initial_population:
                        value=calculate_affinity(target, i)
                        score.append(value)

    if ".txt" in initial_population:
        with open(initial_population, "r") as file:
            score=[]
            for row in file:
                value=calculate_affinity(target, row)
                score.append(value)
    score=score.sort()
    parents=score[:2]
    return parents            
  
def childs(parents=select_parents()):

    '''
    Function to obtain the childs of the parents selected in the select_parents function.

    Parameters:
    -parents: Parents selected in the select_parents function.

    '''

    crossover_point=random.randint(0, len(parents[0])-1)
    child1 = parents[0][:crossover_point] + parents[1][crossover_point:]
    child2 = parents[1][:crossover_point] + parents[0][crossover_point:]
    if Chem.MolFromSmiles(child1) is not None and Chem.MolFromSmiles(child2) is not None:
        return child1, child2
    else:
        childs()

def mutations(smiles, mutation_rate=0.1):

    '''
    Function to mutate a molecule in order to obtain a new molecule with a better affinity to the target.

    Parameters:
    -smiles: Sequence of the molecule in smile format obtained from the childs function.
    -mutation_rate: Probability of mutation of an atom in the molecule.

    '''
    
    molecule=[]
    molecule.append(Chem.MolFromSmiles(smiles[0]))
    molecule.append(Chem.MolFromSmiles(smiles[1]))
    copy=molecule.copy()
    for atom in copy:
        copy[enumerate(atom)]=Chem.rdchem.RWMol(copy[enumerate(atom)])
        if random.uniform(0,1) <= mutation_rate:
            new_atom= random.choice(list(Chem.AtomPDBResidueInfo.GetAtomicSymbolList()))        
            atom.SetAtomicNum(Chem.AtomPDBResidueInfo.GetAtomicNumber(new_atom))
    if Chem.MolToSmiles(copy[enumerate(atom)]) is not None:
        return copy[enumerate(atom)]
    

            
                
                
    


def genetic_algorithm(target="", initial_pop_path=r"", objective_ic50=20, generations=100, bests=10):
    '''
    Function to find the best molecule to bind to a target protein using a genetic algorithm.

    Parameters:
    -target: Sequence of the target protein in fasta format.
    -initial_pop_path: Path of the initial population of smile molecules. If this file does not exist, the function will create it.
    -objective_ic50: Value of the affinity of the target protein that we want to obtain. By default, it is 20.
    -generations: Number of generations of the genetic algorithm. By default, it is 100.
    -bests: Number of best molecules that we want to select from each generation. By default, it is 10.
    '''
    parents=select_parents(initial_population=initial_pop_path, target=target)  
    
    for gen in range(generations):
        new_generation=[]
        for i in range(bests):
            childs=childs(parents=parents)
            mutations=mutations(smiles=childs)
            new_generation.append(mutations)
        parents=new_generation
        score=[]
        for i in new_generation:
            value=calculate_affinity(target, i)
            score.append(value)
        score=score.sort()
        total=zip(parents, score)
        if compare_ic50(list_score=total, objective_ic50=objective_ic50) is not False:
            best_individual, affinity= compare_ic50(list_score=total, objective_ic50=objective_ic50)
            print("Generation:", gen+1)
            print("Best SMILE sequence obtained:", best_individual)
            print("Fitness:", affinity)
            print("--------")
            break
        else:
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
    
