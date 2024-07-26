from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, RWMol
import random
import csv
import numpy as np
import pandas as pd
import time

from affinity_with_target_and_generator import find_candidates
from check_affinity import calculate_affinity


def select_parents(initial_population, target, bests=2):
    """
    Select the parents for the first generations of the genetic algorithm.
    The parents are the molecules with the best affinity to the target.

    Parameters:
        initial_population (str or list): Path of the file with the initial population of molecules 
                                          or a list of SMILES strings.
        target (str): Sequence of the target in FASTA format.
        bests (int, optional): Number of best molecules to select as parents. Defaults to 2.
    
    Returns:
        list of tuples: List of tuples containing the best molecules and their affinities.
    """

    def load_population(file_path):
         
        """
        Load a list of SMILES strings from a CSV file.

        Parameters:
            file_path (str): Path to the CSV file.

        Returns:
            list of str: List of SMILES strings.
        """

        with open(file_path, "r") as file:
            reader = csv.reader(file)
            return [row[0] for row in reader][1:]

    def score_population(population):

        """
        Score a population of SMILES strings based on their affinity to the target.

        Parameters:
            population (list of str): List of SMILES strings.

        Returns:
            list of tuples: List of tuples containing SMILES strings and their affinities, sorted by affinity.
        """

        scores = [calculate_affinity(smile=i, fasta=target) for i in population]
        return sorted(zip(population, scores), key=lambda x: x[1])

    if "generate" in initial_population:
        find_candidates(max_molecules=5, db_smiles=True, target=target, draw_minor=False, generate_qr=False,
                        upload_to_mega=False, arx_db=r"generated_molecules/generated_molecules.txt",
                        accepted_value=3000, name_file_destination="generated_w_algo")
        initial_population = load_population("generated_w_algo.csv")

    elif ".csv" in initial_population or ".txt" in initial_population:
        initial_population = load_population(initial_population)

    else:
        initial_population = [i.replace("@", "").replace("/", "") for i in initial_population]

    total = score_population(initial_population)
    return total[:bests] if total else []


def check_druglikeness(smile):

    """
    Check if a molecule is druglike based on its SMILES string.

    Parameters:
        smile (str): Molecule structure in SMILES format.

    Returns:
        bool: True if the molecule is druglike, False otherwise.
    """

    mol = Chem.MolFromSmiles(smile)
    if mol:
        return (Descriptors.ExactMolWt(mol) < 500 and
                Descriptors.MolLogP(mol) < 5 and
                Descriptors.NumHDonors(mol) < 5 and
                Descriptors.NumHAcceptors(mol) < 10)
    return False


def generate_children(parent1, parent2, mutation_attempts=10):

    """
    Generate two new molecules by crossing two parent SMILES sequences.

    Parameters:
        parent1 (str): SMILES sequence of the first parent.
        parent2 (str): SMILES sequence of the second parent.
        mutation_attempts (int): Number of attempts to generate valid children. Defaults to 10.

    Returns:
        tuple of str: Tuple of two child SMILES sequences or the original parents if valid children cannot be created.
    """

    def crossover(smile1, smile2):
        crossover_point = random.randint(1, min(len(smile1), len(smile2)) - 1)
        child1 = smile1[:crossover_point] + smile2[crossover_point:]
        child2 = smile2[:crossover_point] + smile1[crossover_point:]
        return child1, child2

    child1, child2 = crossover(parent1, parent2)
    if Chem.MolFromSmiles(child1) and Chem.MolFromSmiles(child2):
        return child1, child2

    if mutation_attempts > 0:
        return generate_children(parent1, parent2, mutation_attempts - 1)
    
    print("Failed to generate valid children; returning original parents.")
    return parent1, parent2


def mutate(smile, mutation_rate=0.1):

    """
    Mutate a molecule to obtain new molecule structures.

    Parameters:
        smile (str): SMILES string of the molecule.
        mutation_rate (float): Probability of mutation for each atom in the molecule. Defaults to 0.1.

    Returns:
        list of str: List of mutated SMILES strings.
    """
    atoms = [6, 5, 7, 15, 8, 16, 9, 17, 35, 53]
    aromatic_atoms = [6, 7, 15, 8, 16]

    mol = Chem.MolFromSmiles(smile)
    if not mol:
        return []

    mutated_smiles = set()
    for _ in range(5):
        try:
            mol = Chem.MolFromSmiles(smile)
            rw_mol = RWMol(mol)
            num_atoms = rw_mol.GetNumAtoms()
            for atom_idx in range(num_atoms):
                if random.uniform(0, 1) <= mutation_rate:
                    atom = rw_mol.GetAtomWithIdx(atom_idx)
                    valence = atom.GetTotalValence()
                    if atom.GetIsAromatic():
                        if valence == 2:
                            rw_mol.ReplaceAtom(atom_idx, Chem.Atom(random.choice(aromatic_atoms[1:])))
                        elif valence == 3:
                            rw_mol.ReplaceAtom(atom_idx, Chem.Atom(random.choice(aromatic_atoms[:3])))
                    else:
                        rw_mol.ReplaceAtom(atom_idx, Chem.Atom(random.choice(atoms[:4])))
            Chem.SanitizeMol(rw_mol)
            mutated_smile = Chem.MolToSmiles(rw_mol)
            if check_druglikeness(mutated_smile):
                mutated_smiles.add(mutated_smile)
        except Exception:
            pass

    return list(mutated_smiles)


def prepare_file(file_path, headers=[]):
    '''
    Create a CSV file with specified headers.

    Parameters:
        file_path (str): Path to the file.
        headers (list of str): List of column names.
    '''
    with open(f"{file_path}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)


def genetic_algorithm(target="", initial_pop_path="", objective_ic50=20, generations=100, bests=2,
                      path_save="", save_since=20, name_file_best="", name_img="result", initial="", name_img_initial=""):
    '''
    Find the best molecule to bind to a target protein using a genetic algorithm.

    Parameters:
        target (str): Sequence of the target protein in FASTA format.
        initial_pop_path (str): Path of the initial population of SMILES molecules.
        objective_ic50 (float): Desired IC50 value. Defaults to 20.
        generations (int): Number of generations. Defaults to 100.
        bests (int): Number of best molecules to select. Defaults to 2.
        path_save (str): Path to save the best molecules.
        save_since (float): IC50 value threshold to save molecules. Defaults to 20.
        name_file_best (str): File name to save the best molecules.
        name_img (str): Image file name for the best molecule.
        initial (str): File name for the best molecule from the initial generation.
        name_img_initial (str): Image file name for the initial best molecule.
    '''
    parents = select_parents(initial_population=initial_pop_path, target=target, bests=bests)
    
    best_parent = min(parents, key=lambda x: x[1])
    prepare_file(file_path=path_save, headers=["SMILE", "Affinity"])
    
    with open(f"{initial}.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([best_parent[0], best_parent[1]])
        Draw.MolToImageFile(Chem.MolFromSmiles(best_parent[0]), filename=f"examples/best_molecule_{name_img_initial}.jpg", size=(400, 300))
    
    all_bests = []
    sum_not_improve = 0
    smiles_saved = []
    best_generated = (best_parent[0], best_parent[1])
    
    for gen in range(generations):
        new_generation = []
        
        if sum_not_improve >= 6:
            print("Using child generation due to stagnation.")
            parents = generate_children(smiles_saved[0], smiles_saved[1])
            sum_not_improve = 0
        else:
            parents = [p[0] for p in parents]

        for parent in parents:
            new_generation.extend(mutate(smile=parent, mutation_rate=0.1))

        score = [calculate_affinity(smile=smile, fasta=target) for smile in new_generation]
        total = sorted(zip(new_generation, score), key=lambda x: x[1])
        
        with open(f"{path_save}.csv", "a") as file:
            for item in total:
                if item[1] <= save_since:
                    if item[0] not in pd.read_csv(f"{path_save}.csv").SMILE.tolist():
                        file.write(f"{item[0]}, {item[1]}\n")
        
        best = compare_ic50(list_score=total, objective_ic50=objective_ic50)
        if best or gen == generations - 1:
            best_individual, affinity = best if best else (None, None)
            print(f"Generation: {gen + 1}")
            print(f"Best SMILE sequence obtained: {best_individual}")
            print(f"IC50 value: {affinity}")
            
            if best_individual:
                molecule = Chem.MolFromSmiles(best_individual)
                Draw.MolToImageFile(molecule, filename=f"examples/best_molecule_{name_img}.jpg", size=(400, 300))
                prepare_file(file_path=name_file_best, headers=["SMILE", "Affinity"])
                with open(f"{name_file_best}.csv", "a") as file:
                    file.write(f"{best_individual}, {affinity}\n")
            break
        
        parents = total[:bests]
        
        if gen == 0:
            all_bests.append(float(parents[0][1]))
            min_ic50 = min(all_bests)
            best_generated = (parents[0][0], parents[0][1])
        else:
            min_ic50 = min(all_bests)
            if float(parents[0][1]) >= min_ic50:
                smiles_saved = [x[0] for x in total]
                smiles_saved = [smiles_saved[0], smiles_saved[random.randint(1, len(smiles_saved) - 1)]]
                sum_not_improve += 1
            else:
                if best_generated[1] > parents[0][1]:
                    best_generated = (parents[0][0], parents[0][1])
                sum_not_improve = 0
            if parents[0][1] not in all_bests:
                all_bests.append(float(parents[0][1]))

        print(f"Generation: {gen + 1}")
        print(f"Best SMILE sequence obtained this generation: {parents[0][0]}")
        if parents[0][0] != best_generated[0]:
            print(f"Best SMILE sequence obtained overall: {best_generated[0]}")
        print(f"IC50 value: {parents[0][1]}")
        if float(parents[0][1]) >= min_ic50:
            print("Value not improved")
        time.sleep(4)
    

def compare_ic50(list_score, objective_ic50):
    '''
    Compare the affinity of molecules with the objective IC50 value.

    Parameters:
        list_score (list of tuples): List of tuples containing SMILES and affinity values.
        objective_ic50 (float): Desired IC50 value.

    Returns:
        tuple of str or bool: Tuple of the best SMILES sequence and its affinity if the objective is met, otherwise False.
    '''
    for item in list_score:
        if item[1] <= objective_ic50:
            return item[0], item[1]
    return False
