from rdkit import Chem 
from rdkit.Chem import Draw
from rdkit.Chem import rdMolDescriptors
from check_affinity import calculate_affinity
import random   
import csv
from affinity_with_target_and_generator import find_candidates


def select_parents(initial_population=r"", target=""):
    if not initial_population:
        initial_population=find_candidates(return_path=True,max_molecules=10,db_smiles=False,target=target,draw_minor=False,generate_qr=False,upload_to_mega=False)
    with open(initial_population, "r") as file:
            reader = csv.reader(file)
            initial_population = [row for row in reader[1]]
            score=[]
            for i in initial_population:
                value=calculate_affinity(target, i)
                score.append(value)
            score=score.sort()
            parents=score[:2]
            return parents
  
def childs():
    parents=select_parents()
    crossover_point=random.randint(0, len(parents[0])-1)
    child1 = parents[0][:crossover_point] + parents[1][crossover_point:]
    child2 = parents[1][:crossover_point] + parents[0][crossover_point:]
    if Chem.MolFromSmiles(child1) is not None and Chem.MolFromSmiles(child2) is not None:
        return child1, child2
    else:
        childs()

def mutations(smiles, mutation_rate):
    smiles=childs()
    molecule=[]
    molecule.append(Chem.MolFromSmiles(smiles[0]))
    molecule.append(Chem.MolFromSmiles(smiles[1]))
    copy=molecule.copy()
    for atom in copy[0]:
        if random.uniform(0,1) <= mutation_rate:
            new_atom= random.choice(list(Chem.AtomPDBResidueInfo.GetAtomicSymbolList()))        
            atom.SetAtomicNum(Chem.AtomPDBResidueInfo.GetAtomicNumber(new_atom))
    if Chem.MolToSmiles(copy[0]) is not None:
        return copy[0]
    
def genetic_algorithm(target="", initial_population=r"", mutation_rate=0.1, max_generations=100):
    if not initial_population:
        initial_population=find_candidates(return_path=True,max_molecules=10,db_smiles=False,target=target,draw_minor=False,generate_qr=False,upload_to_mega=False)
    parents=select_parents(initial_population=initial_population, target=target)
    for i in range(max_generations):
        childs()
        mutations(smiles=childs(), mutation_rate=mutation_rate)
        parents=select_parents(initial_population=initial_population, target=target)
        if parents[0][1]<0.1:
            break
    return parents[0]
    
            
                
                
    