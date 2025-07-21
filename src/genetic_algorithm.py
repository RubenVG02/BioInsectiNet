from rdkit import Chem
from rdkit.Chem import Descriptors, BRICS, Draw
import random
import csv
import argparse
import os

from generate_molecules_with_affinity import find_candidates
from check_affinity import predict_affinity, get_best_trial

best_trial = get_best_trial("models/cnn_affinity.db", study_name="cnn_affinity")

LIPINSKI_MAX_MWT = 500
LIPINSKI_MAX_LOGP = 5
LIPINSKI_MAX_HDONORS = 5
LIPINSKI_MAX_HACCEPTORS = 10

DEFAULT_BESTS_TO_SELECT = 2
DEFAULT_MUTATION_ATTEMPTS = 10 # Attempts *per mutation operation*
DEFAULT_MUTATION_RATE = 0.1
DEFAULT_STAGNATION_LIMIT = 6  # Generations without improvement to trigger alternative strategy
DEFAULT_NUM_MUTATIONS_PER_SMILE = 5 # Number of independent mutation attempts on a single SMILE

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a genetic algorithm to optimize SMILES strings for a target sequence.")
    
    # General arguments
    parser.add_argument("--target_fasta", type=str, required=True, help="Target FASTA.")
    parser.add_argument("--generate_smiles_from_optimizer", action="store_true", help="Generate initial population using the optimizer.")
    parser.add_argument("--initial_pop_source", type=str, help="Initial population source (file path or list of SMILES).")
    parser.add_argument("--objective_ic50", type=float, default=50, help="Objective IC50 value for the target.")
    parser.add_argument("--generations", type=int, default=100, help="Number of generations to run.")
    parser.add_argument("--population_size", type=int, default=50, help="Target population size per generation.")
    parser.add_argument("--num_parents_select", type=int, default=DEFAULT_BESTS_TO_SELECT, help="Number of best candidates to select for reproduction.")
    parser.add_argument("--mutation_rate", type=float, default=DEFAULT_MUTATION_RATE, help="Probability of mutation for each atom in the molecule.")
    parser.add_argument("--stagnation_limit", type=int, default=DEFAULT_STAGNATION_LIMIT, help="Generations without improvement to trigger alternative strategy.")
    parser.add_argument("--output_dir", type=str, default="ga_results", help="Directory to save results.")
    parser.add_argument("--all_results_file", type=str, default="all_scored_molecules.csv", help="File to save all scored molecules.")
    parser.add_argument("--best_overall_file", type=str, default="best_molecule_overall.csv", help="File to save the best molecule overall.")
    parser.add_argument("--initial_best_file", type=str, default="best_molecule_initial.csv", help="File to save the best molecule from the initial population.")
    parser.add_argument("--image_dir", type=str, default="molecule_images", help="Subdirectory for images of molecules.")
    parser.add_argument("--mutation_attempts", type=int, default=DEFAULT_MUTATION_ATTEMPTS, help="Number of attempts to generate valid children.")
    parser.add_argument("--generator_output_name_destination", type=str, default="generated_molecules", help="Name of the file to save generated molecules.")

    # Args for generating initial population with optimizer
    parser.add_argument("--generator_ic50_multiplier", type=float, default=20.0, help="Multiplier for the objective IC50 when generating initial population.")
    parser.add_argument("--generator_population_multiplier", type=float, default=15.0, help="Multiplier for the population size when generating initial population.")
    parser.add_argument("--upload_to_mega", action='store_true', help="Upload result CSV to Mega.")
    parser.add_argument("--draw_lowest", action='store_true', help="Save image of molecule with best IC50.")
    parser.add_argument("--db_smiles", action='store_true', help="Use pre-generated SMILES from file.")
    parser.add_argument("--path_db_smiles", type=str, default=r"generated_molecules/generated_molecules.smi", help="Path to SMILES file.")
    parser.add_argument("--generate_qr", action='store_true', help="Generate QR code to Mega link.")
    parser.add_argument("--affinity_model_path", type=str, default=r"models/checkpoints/cnn_affinity/trial_1_loss_0.1974.pth", help="Path to IC50 prediction model.")
    parser.add_argument("--db_affinity_path", type=str, default="models/cnn_affinity.db", help="Path to affinity model DB.")
    parser.add_argument("--study_name", type=str, default="cnn_affinity", help="Name of the study for hyperparameter optimization.")
    parser.add_argument("--generator_model_path", type=str, default=r"models/generator/bindingDB_smiles_filtered_v1.pth", help="Path to RNN generator model.")
    parser.add_argument("--smiles_to_draw", type=int, default=1, help="Number of top SMILES to draw if draw_lowest is True.")

    
    return parser.parse_args()

def load_smiles_from_file(file_path: str) -> list:
    """
    Load a list of SMILES strings from a CSV file.

    Parameters:
        file_path (str): Path to the CSV file.

    Returns:
        list of str: List of SMILES strings.
    """
    smiles_list = []
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()

    if ext == ".smi":
        with open(file_path, "r") as file:
            for line in file:
                smile = line.strip()
                if smile:
                    smiles_list.append(smile)
    else:
        with open(file_path, "r", newline='') as file:
            reader = csv.reader(file)
            try:
                first_row = next(reader)
                # Check if the first row is a header
                if any("SMILES" in col.lower() for col in first_row):
                    pass 
                else:
                    if first_row[0].strip():
                        smiles_list.append(str(first_row[0].strip()))
                for row in reader:
                    if row and row[0].strip():
                        smiles_list.append(str(row[0].strip()))
            except StopIteration:
                pass 

    return smiles_list
    
def score_population(population: list, target: str):
    """
    Score a population of SMILES strings based on their affinity to the target.

    Parameters:
        population (list of str): List of SMILES strings.
        target (str): Sequence of the target in FASTA format.

    Returns:
        list of tuples: List of tuples containing SMILES strings and their affinities, sorted by affinity.
    """
    scores = [10 ** - predict_affinity(smile, target, best_trial=best_trial, path_model=args.affinity_model_path) for smile in population]
    scores = [round(score, 2) for score in scores]
    print(f"[INFO] Scored {len(population)} molecules.")
    return sorted(zip(population, scores), key=lambda x: x[1])

def check_druglikeness(smile: str) -> bool:
    """
    Check if a molecule is druglike based on its SMILES string.

    Parameters:
        smile (str): Molecule structure in SMILES format.

    Returns:
        bool: True if the molecule is druglike, False otherwise.
    """
    mol = Chem.MolFromSmiles(smile)
    if mol:
        return (Descriptors.ExactMolWt(mol) < LIPINSKI_MAX_MWT and
                Descriptors.MolLogP(mol) < LIPINSKI_MAX_LOGP and
                Descriptors.NumHDonors(mol) < LIPINSKI_MAX_HDONORS and
                Descriptors.NumHAcceptors(mol) < LIPINSKI_MAX_HACCEPTORS)
    return False

def generate_children(parent1: str, parent2: str, mutation_attempts: int = DEFAULT_MUTATION_ATTEMPTS) -> list[str]:
    """
    Generate two new molecules by crossing two parent SMILES sequences.

    Parameters:
        parent1 (str): SMILES sequence of the first parent.
        parent2 (str): SMILES sequence of the second parent.
        mutation_attempts (int): Number of attempts to generate valid children. Defaults to 10.

    Returns:
        list of str: List of two child SMILES sequences or the original parents if valid children cannot be created.
    """

    from rdkit.Chem import BRICS

    def crossover(smile1: str, smile2: str) -> tuple[str, str]:
        mol1 = Chem.MolFromSmiles(smile1)
        mol2 = Chem.MolFromSmiles(smile2)
        if not mol1 or not mol2:
            return smile1, smile2

        # Fragment both molecules
        frags1 = [frag for frag in BRICS.BRICSDecompose(mol1)]
        frags2 = [frag for frag in BRICS.BRICSDecompose(mol2)]

        if not frags1 or not frags2:
            return smile1, smile2

        # Select random fragments from each
        frag1 = Chem.MolFromSmiles(random.choice(list(frags1)))
        frag2 = Chem.MolFromSmiles(random.choice(list(frags2)))

        # Combine fragments to form children
        # (BRICS.BRICSBuild returns a generator)
        try:
            child1 = next(BRICS.BRICSBuild([frag1, frag2]))
            child2 = next(BRICS.BRICSBuild([frag2, frag1]))
            return child1, child2
        except StopIteration:
            return smile1, smile2


    child1, child2 = crossover(parent1, parent2)
    child1 = Chem.MolToSmiles(child1) if isinstance(child1, Chem.Mol) else child1
    child2 = Chem.MolToSmiles(child2) if isinstance(child2, Chem.Mol) else child2
    if Chem.MolFromSmiles(child1) is  Chem.MolFromSmiles(child2):
        print(f"Failed to crossover {parent1} and {parent2}. Returning parents.")
        return [parent1, parent2]
    

    if mutation_attempts > 0:
        return generate_children(parent1, parent2, mutation_attempts - 1)
    
    return [parent1, parent2]


from rdkit.Chem import BRICS

def mutate(smile: str, num_mutations: int = DEFAULT_NUM_MUTATIONS_PER_SMILE) -> list[str]:
    mol = Chem.MolFromSmiles(smile)
    if not mol:
        return []

    mutated_smiles = set()
    fragments = list(BRICS.BRICSDecompose(mol))

    if not fragments:
        return []

    for _ in range(num_mutations):
        try:
            # In order to create a mutation, we will remove a random fragment and add a new one
            frag_to_remove = random.choice(fragments)
            fragments_copy = fragments.copy()
            fragments_copy.remove(frag_to_remove)

            # Add a new fragment
            random_small_frag = Chem.MolFromSmiles(random.choice(['C', 'CC', 'N', 'O']))  # Fragmentos simples
            if random_small_frag:
                # Create a new molecule by combining the remaining fragments with the new fragment
                new_mol_generator = BRICS.BRICSBuild([Chem.MolFromSmiles(f) for f in fragments_copy] + [random_small_frag])
                new_smile = next(new_mol_generator, None)
                if new_smile:
                    smile_str = Chem.MolToSmiles(new_smile)
                    if smile_str and check_druglikeness(smile_str):
                        mutated_smiles.add(smile_str)
        except Exception:
            continue

    return list(mutated_smiles)


def prepare_file(file_path: str, headers: list[str] = []) -> None:
    """
    Create a CSV file with specified headers.

    Parameters:
        file_path (str): Path to the file.
        headers (list of str): List of column names.
    """
    with open(f"{file_path}.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(headers)

def save_output(file_path: str, data: list[tuple[str, float]]) -> None:
    """
    Save the output to a CSV file.

    Parameters:
        file_path (str): Path to the file.
        data (list of tuples): List of tuples containing SMILES strings and their affinities.
    """
    with open(f"{file_path}.csv", "a", newline="") as file:
        writer = csv.writer(file)
        for row in data:
            writer.writerow(row)


def genetic_algorithm(
    target_fasta: str,
    initial_pop_source: list[str],
    objective_ic50: float = 20.0,
    generations: int = 100,
    population_size: int = 50,
    num_parents_select: int = DEFAULT_BESTS_TO_SELECT,
    mutation_rate: float = DEFAULT_MUTATION_RATE,
    stagnation_limit: int = DEFAULT_STAGNATION_LIMIT,
    output_dir: str = "ga_results",
    all_results_file: str = "all_scored_molecules",
    best_overall_file: str = "best_molecule_overall",
    initial_best_file: str = "best_molecule_initial",
    image_dir: str = "molecule_images"
):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, image_dir), exist_ok=True)

    if len(initial_pop_source) > population_size:
        print(f"[INFO] Initial population size ({len(initial_pop_source)}) exceeds the target population size ({population_size}). Trimming to {population_size}.")
        population = random.sample(initial_pop_source, population_size)
    scored_population = score_population(population, target_fasta)

    best_initial = scored_population[0]
    save_output(os.path.join(output_dir, initial_best_file), [best_initial])

    best_overall = best_initial
    generations_without_improvement = 0

    def tournament_selection(scored_population, num_parents, tournament_size=3):
            selected_parents = []
            for _ in range(num_parents):
                tournament = random.sample(scored_population, tournament_size)
                winner = min(tournament, key=lambda x: x[1])  # select the one with the lowest affinity
                selected_parents.append(winner[0])  # get the SMILES of the winner
            return selected_parents

    for gen in range(generations):
        print(f"Generation {gen+1}/{generations}")
        
        if best_overall[1] <= objective_ic50:
            print(f"Objective IC50 {objective_ic50} reached with {best_overall[1]:.2f}. Stopping.")
            break

        parents = tournament_selection(scored_population, num_parents_select, tournament_size=3)

        children = []
        while len(children) < population_size:
            p1, p2 = random.sample(parents, 2)
            children.extend(generate_children(p1, p2))

        mutated_children = []
        for child in children:
            if random.random() < mutation_rate:
                mutated = mutate(child)
                if mutated:
                    mutated_children.extend(mutated)
            else:
                mutated_children.append(child)
           

        next_gen_candidates = [
            mol for mol in mutated_children
            if Chem.MolFromSmiles(mol) and check_druglikeness(mol)
        ]

        if len(next_gen_candidates) < population_size:
            next_gen_candidates.extend(parents)
            next_gen_candidates = next_gen_candidates[:population_size]

        scored_population = score_population(next_gen_candidates, target_fasta)
        scored_population = list(set(scored_population))  # Remove duplicates
        scored_population = sorted(scored_population, key=lambda x: x[1])

        save_output(os.path.join(output_dir, all_results_file), scored_population)

        if scored_population[0][1] < best_overall[1]:
            best_overall = scored_population[0]
            generations_without_improvement = 0
            print(f"New best: {best_overall}")
        else:
            generations_without_improvement += 1
            print(f"No improvement. Stagnation: {generations_without_improvement}/{stagnation_limit}. Current best: {best_overall}")

        if generations_without_improvement >= stagnation_limit:
            print("Stagnation limit reached. Stopping.")
            save_output(os.path.join(output_dir, best_overall_file), [best_overall])
            break

    print(f"Best molecule found: {best_overall}")
    if image_dir:
        best_mol = Chem.MolFromSmiles(best_overall[0])
        if best_mol:
            img_path = os.path.join(output_dir, image_dir, f"best_molecule_gen{gen+1}.png")
            Draw.MolToFile(best_mol, img_path)
            print(f"Image of the best molecule saved to {img_path}")
    return best_overall
  
if __name__ == "__main__":
    args = parse_arguments()
    if os.path.exists(os.path.join("output_ga", args.output_dir)):
        i = 1
        while os.path.exists(os.path.join("output_ga", f"{args.output_dir}_{i}")):
            i += 1
        print(f"[INFO] Output directory already exists. Creating a new one: {args.output_dir}_{i}")
        output_dir = os.path.join("output_ga", f"{args.output_dir}_{i}")
    else:
        print(f"[INFO] Creating output directories: {args.output_dir} and {args.image_dir}")
        output_dir = os.path.join("output_ga", args.output_dir)
    image_dir = os.path.join(output_dir, args.image_dir)
    os.makedirs("output_ga", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    
    # Prepare output files
    prepare_file(os.path.join(output_dir, args.all_results_file), headers=["SMILES", "Affinity"])
    prepare_file(os.path.join(output_dir, args.best_overall_file), headers=["SMILES", "Affinity"])
    prepare_file(os.path.join(output_dir, args.initial_best_file), headers=["SMILES", "Affinity"])
    
    if args.generate_smiles_from_optimizer and args.initial_pop_source is not None:
        raise ValueError("Cannot specify both --generate_smiles_from_optimizer and --initial_pop_source.")

    if args.generate_smiles_from_optimizer:
        # Generate initial population using the optimizer
        # In order to save time, the objective IC50 is set to a higher value than the one used in the genetic algorithm
        print("[INFO] Generating initial population using the optimizer...")
        find_candidates(
            target=args.target_fasta,
            name_file_destination=args.generator_output_name_destination,
            max_molecules=args.population_size,
            total_generated=int(args.population_size * args.generator_population_multiplier),
            accepted_value=args.objective_ic50 * args.generator_ic50_multiplier,
            upload_to_mega=args.upload_to_mega,
            draw_lowest=args.draw_lowest,
            smiles_to_draw=args.smiles_to_draw,
            db_smiles=args.db_smiles,
            path_db_smiles=args.path_db_smiles,
            generate_qr=args.generate_qr,
            affinity_model_path=args.affinity_model_path,
            db_affinity_path=args.db_affinity_path,
            study_name=args.study_name,
            generator_model_path=args.generator_model_path,
            output_dir=output_dir    
        )

        initial_population = load_smiles_from_file(os.path.join(output_dir, args.generator_output_name_destination, args.generator_output_name_destination + ".csv"))

    elif args.initial_pop_source and os.path.isfile(args.initial_pop_source):
        print(f"[INFO] Loading initial population from {args.initial_pop_source}...")
        initial_population = load_smiles_from_file(args.initial_pop_source)
    else:
        print("[INFO] Using provided initial population source as a list of SMILES.")
        initial_population = args.initial_pop_source.split(',')

    # Run the genetic algorithm
    genetic_algorithm(
        target_fasta=args.target_fasta,
        initial_pop_source=initial_population,
        objective_ic50=args.objective_ic50,
        generations=args.generations,
        population_size=args.population_size,
        num_parents_select=args.num_parents_select,
        mutation_rate=args.mutation_rate,
        stagnation_limit=args.stagnation_limit,
        output_dir=output_dir,
        all_results_file=args.all_results_file,
        best_overall_file=args.best_overall_file,
        initial_best_file=args.initial_best_file,
        image_dir=args.image_dir
    )