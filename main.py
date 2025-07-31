import argparse
import os
import sys
import time
import json
from datetime import datetime
from typing import List, Dict, Any

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from generation_RNN import generate_druglike_molecules, load_unique_chars_dict
from generate_molecules_with_affinity import find_candidates
from genetic_algorithm import genetic_algorithm, load_smiles_from_file
from docking import fasta_to_pdb_esmfold, run_docking_vina
from docking import visualize_docking_results, viability_check
from docking import prepare_ligand_meeko, prepare_receptor_meeko
from docking import prepare_ligand_obabel, prepare_receptor_obabel
from docking import prepare_ligand_rdkit_manual
from docking import extract_affinities_from_vina_output
from docking import compute_affinity_statistics, plot_affinity_histogram

from src.utils.logger import get_logger, set_log_file, log_info, log_warning, log_error, log_success


class BioInsectiNetPipeline:
    """
    Main pipeline class for all the steps in BioInsectiNet.
    """
    
    def __init__(self, args):
        """Initialize the pipeline with default configurations."""
        self.args = args
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {}
        self.output_base_dir = f"bioinsectinet_results_{self.session_id}"
        os.makedirs(self.output_base_dir, exist_ok=True)
        
        # Log file for the session
        self.log_file = os.path.join(self.output_base_dir, f"pipeline_log_{self.session_id}.txt")
        
        self.logger = get_logger("BioInsectiNet", self.log_file)
        set_log_file(self.log_file)  
        
    def log(self, message: str, level: str = "INFO"):
        """Log message using unified logging system."""
        if level == "INFO":
            self.logger.info(message)
        elif level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)
        elif level == "SUCCESS":
            self.logger.success(message)
        else:
            self.logger.info(message)
    
    def save_session_config(self, args):
        config_file = os.path.join(self.output_base_dir, "session_config.json")
        config = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "arguments": vars(args),
            "pipeline_steps": {
                "generation": args.run_generation,
                "affinity_filtering": args.run_affinity,
                "genetic_algorithm": args.run_ga,
                "docking": args.run_docking
            }
        }
        
        with open(config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        self.log(f"Session configuration saved to {config_file}")
    
    def step_1_molecule_generation(self, args) -> List[str]:
        """
        Step 1: Generate molecules using RNN models.
        
        Returns:
            List[str]: List of generated SMILES strings
        """
        if not args.run_generation:
            self.log("Skipping molecule generation step")
            if args.input_smiles_file:
                self.log(f"Loading existing SMILES from {args.input_smiles_file}")
                return load_smiles_from_file(args.input_smiles_file)
            else:
                raise ValueError("No SMILES source provided. Either enable generation or provide input file.")
        
        self.log("=" * 60)
        self.log("STEP 1: MOLECULE GENERATION")
        self.log("=" * 60)
        
        step_start = time.time()
        
        try:
            generation_dir = os.path.join(self.output_base_dir, "01_generation")
            os.makedirs(generation_dir, exist_ok=True)
            
            unique_chars_dict = load_unique_chars_dict(args.unique_chars_dict_path)
            
            base_name = os.path.basename(args.generator_model_path)
            data_path = base_name.replace("_v1.pth", ".txt").replace("_v2.pth", ".txt")
            
            key_found = next((key for key in unique_chars_dict if key.endswith(data_path)), None)
            if not key_found:
                raise ValueError(f"Unique characters for {data_path} not found in dictionary")
            
            unique_chars = unique_chars_dict[key_found]
            char_to_idx = {char: idx for idx, char in enumerate(unique_chars)}
            
            self.log(f"Using generator model: {args.generator_model_path}")
            self.log(f"Generating {args.total_generated} molecules")
            
            generated_smiles = generate_druglike_molecules(
                model_path=args.generator_model_path,
                char_to_idx=char_to_idx,
                vocab_size=len(unique_chars),
                num_molecules=args.total_generated,
                min_length=args.min_length,
                max_length=args.max_length,
                temperature=args.temperature,
                save_images=False,
                output_dir=generation_dir
            )
            
            smiles_file = os.path.join(generation_dir, "generated_molecules.smi")
            with open(smiles_file, "w") as f:
                for smile in generated_smiles:
                    f.write(f"{smile}\n")
            
            step_time = time.time() - step_start
            self.log(f"Generated {len(generated_smiles)} molecules in {step_time:.2f} seconds")
            self.results["generation"] = {
                "num_generated": len(generated_smiles),
                "output_file": smiles_file,
                "time": step_time
            }
            
            return generated_smiles
            
        except Exception as e:
            self.log(f"ERROR in molecule generation: {e}", "ERROR")
            raise
    
    def step_2_affinity_filtering(self, args, generated_smiles: List[str]) -> List[str]:
        """
        Step 2: Filter molecules by predicted affinity to target protein.
        
        Args:
            generated_smiles: List of SMILES from generation step
            
        Returns:
            List[str]: List of filtered SMILES with good affinity
        """
        if not args.run_affinity:
            self.log("Skipping affinity filtering step")
            return generated_smiles[:args.max_molecules]  # Just take first N molecules
        
        self.log("=" * 60)
        self.log("STEP 2: AFFINITY FILTERING")
        self.log("=" * 60)
        
        step_start = time.time()
        
        try:
            affinity_dir = os.path.join(self.output_base_dir, "02_affinity_filtering")
            os.makedirs(affinity_dir, exist_ok=True)
            
            # Save generated SMILES to temporary file for processing
            temp_smiles_file = os.path.join(affinity_dir, "temp_generated.smi")
            with open(temp_smiles_file, "w") as f:
                for smile in generated_smiles:
                    f.write(f"{smile}\n")
            
            self.log(f"Filtering {len(generated_smiles)} molecules against target protein")
            self.log(f"Target IC50 threshold: {args.accepted_value}")
            
            # Run affinity filtering
            find_candidates(
                target=args.target_fasta,
                name_file_destination="filtered_candidates",
                output_dir=affinity_dir,
                upload_to_mega=args.upload_to_mega,
                draw_lowest=args.draw_lowest,
                max_molecules=args.max_molecules,
                db_smiles=True,  # Use the temp file
                path_db_smiles=temp_smiles_file,
                accepted_value=args.accepted_value,
                generate_qr=args.generate_qr,
                affinity_model_path=args.affinity_model_path,
                generator_model_path=args.generator_model_path,
                db_affinity_path=args.db_affinity_path,
                study_name=args.study_name,
                total_generated=len(generated_smiles),
                smiles_to_draw=args.smiles_to_draw
            )
            
            results_file = os.path.join(affinity_dir, "filtered_candidates", "filtered_candidates.csv")
            filtered_smiles = load_smiles_from_file(results_file)
            
            # Clean up temp file
            os.remove(temp_smiles_file)
            
            step_time = time.time() - step_start
            self.log(f"Filtered to {len(filtered_smiles)} molecules with good affinity in {step_time:.2f} seconds")
            self.results["affinity_filtering"] = {
                "num_input": len(generated_smiles),
                "num_filtered": len(filtered_smiles),
                "output_file": results_file,
                "time": step_time
            }
            
            return filtered_smiles
            
        except Exception as e:
            self.log(f"ERROR in affinity filtering: {e}", "ERROR")
            raise
    
    def step_3_genetic_algorithm(self, args, filtered_smiles: List[str]) -> Dict[str, Any]:
        """
        Step 3: Optimize molecules using genetic algorithm.
        
        Args:
            filtered_smiles: List of SMILES from affinity filtering
            
        Returns:
            Dict: GA results including best molecule
        """
        if not args.run_ga:
            self.log("Skipping genetic algorithm optimization step")
            # Return the best molecule from filtering (lowest IC50)
            if filtered_smiles:
                return {
                    "best_molecule": (filtered_smiles[0], 0.0),  # Placeholder IC50
                    "optimization_performed": False
                }
            else:
                raise ValueError("No molecules available for GA step")
        
        self.log("=" * 60)
        self.log("STEP 3: GENETIC ALGORITHM OPTIMIZATION")
        self.log("=" * 60)
        
        step_start = time.time()
        
        try:
            ga_dir = os.path.join(self.output_base_dir, "03_genetic_algorithm")
            os.makedirs(ga_dir, exist_ok=True)
            
            self.log(f"Starting GA optimization with {len(filtered_smiles)} initial molecules")
            self.log(f"Target IC50: {args.objective_ic50}")
            self.log(f"Generations: {args.generations}")
            self.log(f"Population size: {args.population_size}")
            
            # Run genetic algorithm
            best_result = genetic_algorithm(
                target_fasta=args.target_fasta,
                initial_pop_source=filtered_smiles,
                objective_ic50=args.objective_ic50,
                generations=args.generations,
                population_size=args.population_size,
                num_parents_select=args.num_parents_select,
                mutation_rate=args.mutation_rate,
                stagnation_limit=args.stagnation_limit,
                output_dir=ga_dir,
                all_results_file="all_scored_molecules.csv",
                best_overall_file="best_molecule_overall.csv",
                initial_best_file="best_molecule_initial.csv",
                image_dir="molecule_images",
                affinity_model_path=args.affinity_model_path
            )

            step_time = time.time() - step_start
            self.log(f"GA optimization completed in {step_time:.2f} seconds")
            self.log(f"Best molecule: {best_result[0]} with IC50: {best_result[1]}")
            
            self.results["genetic_algorithm"] = {
                "best_smiles": best_result[0],
                "best_ic50": best_result[1],
                "initial_population_size": len(filtered_smiles),
                "optimization_performed": True,
                "time": step_time
            }
            
            return {
                "best_molecule": best_result,
                "optimization_performed": True
            }
            
        except Exception as e:
            self.log(f"ERROR in genetic algorithm: {e}", "ERROR")
            raise
    
    def step_4_molecular_docking(self, args, ga_results: Dict[str, Any] = None, filtered_smiles: List[str] = None) -> Dict[str, Any]:
        """
        Step 4: Perform molecular docking analysis.
        
        Args:
            ga_results: Results from genetic algorithm step (optional)
            filtered_smiles: List of filtered SMILES for docking (optional)
            
        Returns:
            Dict: Docking results
        """
        if not args.run_docking:
            self.log("Skipping molecular docking step")
            return {"docking_performed": False}
        
        self.log("=" * 60)
        self.log("STEP 4: MOLECULAR DOCKING")
        self.log("=" * 60)
        
        step_start = time.time()
        
        try:
            docking_dir = os.path.join(self.output_base_dir, "04_docking")
            os.makedirs(docking_dir, exist_ok=True)
            
            # Determine molecules to dock
            molecules_to_dock = []
            
            if ga_results and ga_results.get("optimization_performed", False):
                # GA was executed - use GA optimized molecule plus top N-1 from filtering
                best_molecule = ga_results["best_molecule"]
                molecules_to_dock.append((best_molecule[0], best_molecule[1]))  # GA best molecule
                
                # Add additional molecules from filtering if requested and available
                if args.num_molecules_docking > 1 and filtered_smiles:
                    remaining_slots = args.num_molecules_docking - 1
                    additional_molecules = min(remaining_slots, len(filtered_smiles))
                    for i in range(additional_molecules):
                        # Skip GA molecule if it's already in filtered_smiles
                        if filtered_smiles[i] != best_molecule[0]:
                            molecules_to_dock.append((filtered_smiles[i], 0.0))
                        elif i + 1 < len(filtered_smiles):
                            molecules_to_dock.append((filtered_smiles[i + 1], 0.0))
                
                self.log(f"Using GA-optimized molecule + top {len(molecules_to_dock)-1} from filtering for docking")
                self.log(f"GA best molecule: {best_molecule[0]} (IC50: {best_molecule[1]:.2f})")
                
            elif filtered_smiles:
                # Use top N molecules from affinity filtering (GA not executed or skipped)
                num_molecules = min(args.num_molecules_docking, len(filtered_smiles))
                molecules_to_dock = [(smile, 0.0) for smile in filtered_smiles[:num_molecules]]  # Placeholder IC50
                self.log(f"Using top {num_molecules} molecules from affinity filtering for docking")
                
            elif ga_results and not ga_results.get("optimization_performed", False):
                # GA was skipped but we have the best molecule from filtering
                best_molecule = ga_results["best_molecule"]
                # Still respect num_molecules_docking parameter even when GA was skipped
                num_molecules = min(args.num_molecules_docking, 1)  # At least use the GA best molecule
                molecules_to_dock = [(best_molecule[0], best_molecule[1])]
                self.log(f"Using best molecule from filtering (GA was skipped): {best_molecule[0]}")
                
            else:
                raise ValueError("No molecules available for docking. Either run GA, affinity filtering, or provide molecules.")
            
            self.log(f"Performing docking on {len(molecules_to_dock)} molecule(s)")
            self.log(f"Against target protein (FASTA provided)")
            
            # Generate receptor PDB from FASTA (only once)
            self.log("Generating 3D structure from FASTA sequence...")
            pdb_file = fasta_to_pdb_esmfold(args.target_fasta, docking_dir)
            
            # Prepare receptor (only once) with fallback system
            self.log("Preparing receptor for docking...")
            receptor_pdbqt = os.path.join(docking_dir, "receptor.pdbqt")
            
            # Try different methods in order of preference
            methods_tried = []
            receptor_prepared = False
            
            # Try Meeko first
            if prepare_receptor_meeko(pdb_file, receptor_pdbqt):
                receptor_prepared = True
                self.log("Receptor prepared successfully with Meeko")
            else:
                methods_tried.append("Meeko")
                self.log("Meeko receptor preparation failed, trying Open Babel...", "WARNING")
                
                # Try Open Babel as fallback
                if prepare_receptor_obabel(pdb_file, receptor_pdbqt):
                    receptor_prepared = True
                    self.log("Receptor prepared successfully with Open Babel")
                else:
                    methods_tried.append("Open Babel")
            
            if not receptor_prepared:
                raise RuntimeError(f"Failed to prepare receptor with all methods tried: {', '.join(methods_tried)}")
            
            # Dock each molecule
            docking_results_list = []
            
            for i, (smiles, ic50) in enumerate(molecules_to_dock):
                self.log(f"Docking molecule {i+1}/{len(molecules_to_dock)}: {smiles}")
                
                # Create subdirectory for this molecule
                mol_dir = os.path.join(docking_dir, f"molecule_{i+1}")
                os.makedirs(mol_dir, exist_ok=True)
                
                # Prepare ligand with three-tier fallback system
                ligand_pdbqt = os.path.join(mol_dir, f"ligand_{i+1}.pdbqt")
                ligand_prepared = False
                ligand_methods_tried = []
                
                if prepare_ligand_meeko(smiles, ligand_pdbqt):
                    ligand_prepared = True
                    self.log(f"Ligand {i+1} prepared successfully with Meeko")
                else:
                    ligand_methods_tried.append("Meeko")
                    self.log(f"Meeko ligand preparation failed for molecule {i+1}, trying Open Babel...", "WARNING")
                    

                    if prepare_ligand_obabel(smiles, ligand_pdbqt):
                        ligand_prepared = True
                        self.log(f"Ligand {i+1} prepared successfully with Open Babel")
                    else:
                        ligand_methods_tried.append("Open Babel")
                        self.log(f"Open Babel ligand preparation failed for molecule {i+1}, trying RDKit manual...", "WARNING")
                        
                        if prepare_ligand_rdkit_manual(smiles, ligand_pdbqt):
                            ligand_prepared = True
                            self.log(f"Ligand {i+1} prepared successfully with RDKit manual method")
                        else:
                            ligand_methods_tried.append("RDKit Manual")
                
                if not ligand_prepared:
                    self.log(f"Failed to prepare ligand {i+1} with all methods tried ({', '.join(ligand_methods_tried)}): {smiles}", "WARNING")
                    continue
                
                self.log(f"Running molecular docking for molecule {i+1}...")
                try:
                    docking_output = run_docking_vina(
                        ligand_pdbqt=ligand_pdbqt,
                        receptor_pdbqt=receptor_pdbqt,
                        center=args.docking_center,
                        box_size=args.docking_box_size,
                        output_dir=mol_dir,
                        vina_path=args.vina_path
                    )
                    
                    docked_file = os.path.join(mol_dir, "docked.pdbqt")
                    affinities = []
                    viability_result = None
                    
                    if os.path.exists(docked_file):
                        affinities = extract_affinities_from_vina_output(docked_file)
                        
                        if affinities:
                            stats = compute_affinity_statistics(affinities)
                            self.log(f"Molecule {i+1} - Best affinity: {min(affinities):.2f} kcal/mol")
                            self.log(f"Molecule {i+1} - Average affinity: {stats['mean']:.2f} kcal/mol")
                            
                            histogram_file = os.path.join(mol_dir, f"affinity_histogram_mol_{i+1}.png")
                            plot_affinity_histogram(affinities, histogram_file)
                            self.log(f"Affinity histogram saved: {histogram_file}")
                            
                            viability_result = viability_check(min_affinity=-7.0, max_affinity=0.0, stats=stats)
                            
                            try:
                                visualize_docking_results(docked_file, receptor_pdbqt, mol_dir)
                                pse_file = os.path.join(mol_dir, "docking_results.pse")
                                if os.path.exists(pse_file):
                                    self.log(f"PyMOL visualization saved: {pse_file}")
                                else:
                                    self.log(f"PyMOL visualization attempted but file not found", "WARNING")
                            except Exception as viz_e:
                                self.log(f"Visualization failed for molecule {i+1}: {viz_e}", "WARNING")
                        else:
                            self.log(f"No affinities found for molecule {i+1}", "WARNING")
                    
                    docking_results_list.append({
                        "molecule_id": i+1,
                        "smiles": smiles,
                        "predicted_ic50": ic50,
                        "ligand_file": ligand_pdbqt,
                        "output_dir": mol_dir,
                        "docked_file": docked_file if os.path.exists(docked_file) else None,
                        "affinities": affinities,
                        "best_affinity": min(affinities) if affinities else None,
                        "viability": viability_result,
                        "docking_success": True
                    })
                    
                    self.log(f"Docking completed for molecule {i+1}")
                    
                except Exception as e:
                    self.log(f"Docking failed for molecule {i+1}: {e}", "WARNING")
                    docking_results_list.append({
                        "molecule_id": i+1,
                        "smiles": smiles,
                        "predicted_ic50": ic50,
                        "ligand_file": ligand_pdbqt,
                        "output_dir": mol_dir,
                        "docking_success": False,
                        "error": str(e)
                    })
            
            summary_file = os.path.join(docking_dir, "docking_summary.json")
            with open(summary_file, "w") as f:
                json.dump({
                    "total_molecules": len(molecules_to_dock),
                    "successful_dockings": sum(1 for r in docking_results_list if r.get("docking_success", False)),
                    "failed_dockings": sum(1 for r in docking_results_list if not r.get("docking_success", False)),
                    "results": docking_results_list
                }, f, indent=2)
            
            step_time = time.time() - step_start
            successful_dockings = sum(1 for r in docking_results_list if r.get("docking_success", False))
            self.log(f"Docking completed in {step_time:.2f} seconds")
            self.log(f"Successful dockings: {successful_dockings}/{len(molecules_to_dock)}")
            
            docking_results = {
                "docking_performed": True,
                "total_molecules": len(molecules_to_dock),
                "successful_dockings": successful_dockings,
                "failed_dockings": len(molecules_to_dock) - successful_dockings,
                "receptor_file": receptor_pdbqt,
                "output_dir": docking_dir,
                "summary_file": summary_file,
                "detailed_results": docking_results_list,
                "time": step_time
            }
            
            self.results["docking"] = docking_results
            return docking_results
            
        except Exception as e:
            self.log(f"ERROR in molecular docking: {e}", "ERROR")
            raise
    
    def generate_final_report(self, args):
        """Generate a comprehensive final report of the pipeline run."""
        report_file = os.path.join(self.output_base_dir, "PIPELINE_REPORT.md")
        
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("# BioInsectiNet Pipeline Report\n\n")
            f.write(f"**Session ID:** {self.session_id}\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Target Protein:** {args.target_fasta[:50]}...\n\n")
            
            f.write("## Pipeline Configuration\n\n")
            f.write(f"- **Generation:** {'YES' if args.run_generation else 'NO'}\n")
            f.write(f"- **Affinity Filtering:** {'YES' if args.run_affinity else 'NO'}\n")
            f.write(f"- **Genetic Algorithm:** {'YES' if args.run_ga else 'NO'}\n")
            f.write(f"- **Molecular Docking:** {'YES' if args.run_docking else 'NO'}\n\n")
            
            f.write("## Results Summary\n\n")
            
            total_time = sum(step.get("time", 0) for step in self.results.values())
            f.write(f"**Total Pipeline Time:** {total_time:.2f} seconds\n\n")
            
            for step_name, step_results in self.results.items():
                f.write(f"### {step_name.replace('_', ' ').title()}\n")
                for key, value in step_results.items():
                    if key != "time":
                        f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
                f.write(f"- **Execution Time:** {step_results.get('time', 0):.2f} seconds\n\n")
            
            if "genetic_algorithm" in self.results:
                ga_results = self.results["genetic_algorithm"]
                f.write("## Best Molecule Found\n\n")
                f.write(f"**SMILES:** `{ga_results['best_smiles']}`\n")
                f.write(f"**Predicted IC50:** {ga_results['best_ic50']:.2f}\n")
                f.write(f"**Optimization:** {'Yes' if ga_results['optimization_performed'] else 'No'}\n\n")
            
            if "docking" in self.results and self.results["docking"]["docking_performed"]:
                docking_results = self.results["docking"]
                f.write("## Docking Results\n\n")
                f.write(f"**Total Molecules Docked:** {docking_results['total_molecules']}\n")
                f.write(f"**Successful Dockings:** {docking_results['successful_dockings']}\n")
                f.write(f"**Failed Dockings:** {docking_results['failed_dockings']}\n")
                f.write(f"**Success Rate:** {(docking_results['successful_dockings']/docking_results['total_molecules']*100):.1f}%\n\n")
                
                successful_molecules = [r for r in docking_results['detailed_results'] if r.get('docking_success', False)]
                if successful_molecules:
                    f.write("### Successfully Docked Molecules\n\n")
                    for mol in successful_molecules:
                        f.write(f"#### Molecule {mol['molecule_id']}\n")
                        f.write(f"- **SMILES:** `{mol['smiles']}`\n")
                        f.write(f"- **Predicted IC50:** {mol['predicted_ic50']:.2f}\n")
                        if mol.get('best_affinity'):
                            f.write(f"- **Best Binding Affinity:** {mol['best_affinity']:.2f} kcal/mol\n")
                        if mol.get('affinities'):
                            f.write(f"- **Number of Poses:** {len(mol['affinities'])}\n")
                        if mol.get('viability') is not None:
                            viability_text = "Good" if mol['viability'] else "Poor"
                            f.write(f"- **Viability Assessment:** {viability_text}\n")
                        f.write(f"- **Output Directory:** `{mol['output_dir']}`\n")
                        pse_file = os.path.join(mol['output_dir'], "docking_results.pse")
                        if os.path.exists(pse_file):
                            f.write(f"- **PyMOL Visualization:** `{pse_file}`\n")
                        f.write("\n")
                    f.write("\n")
            
            f.write("## Output Files\n\n")
            f.write(f"All results are saved in: `{self.output_base_dir}`\n\n")
            
            f.write("### Directory Structure\n")
            f.write("```\n")
            f.write(f"{self.output_base_dir}/\n")
            if args.run_generation:
                f.write("├── 01_generation/\n")
            if args.run_affinity:
                f.write("├── 02_affinity_filtering/\n")
            if args.run_ga:
                f.write("├── 03_genetic_algorithm/\n")
            if args.run_docking:
                f.write("├── 04_docking/\n")
            f.write(f"├── pipeline_log_{self.session_id}.txt\n")
            f.write("├── session_config.json\n")
            f.write("└── PIPELINE_REPORT.md\n")
            f.write("```\n")
        
        self.log(f"Final report generated: {report_file}")
    
    def run_pipeline(self, args):
        """Run the complete pipeline based on user configuration."""
        self.log("Starting BioInsectiNet Pipeline")
        self.log(f"Session ID: {self.session_id}")
        self.log(f"Output directory: {self.output_base_dir}")
        
        self.save_session_config(args)
        
        pipeline_start = time.time()
        
        try:
            # Step 1: Molecule Generation
            generated_smiles = self.step_1_molecule_generation(args)
            
            # Step 2: Affinity Filtering
            filtered_smiles = self.step_2_affinity_filtering(args, generated_smiles)
            
            # Step 3: Genetic Algorithm Optimization
            ga_results = self.step_3_genetic_algorithm(args, filtered_smiles)
            
            # Step 4: Molecular Docking
            docking_results = self.step_4_molecular_docking(args, ga_results, filtered_smiles)
            
            # Generate final report
            total_time = time.time() - pipeline_start
            self.log("=" * 60)
            self.log("\033[92mPIPELINE COMPLETED SUCCESSFULLY\033[0m")
            self.log(f"Total execution time: {total_time:.2f} seconds")
            self.log("=" * 60)
            
            self.generate_final_report(args)
            
            return {
                "success": True,
                "session_id": self.session_id,
                "output_dir": self.output_base_dir,
                "results": self.results
            }
            
        except Exception as e:
            self.log(f"\033[91mPipeline execution failed: {e}\033[0m", "ERROR")
            return {
                "success": False,
                "error": str(e),
                "session_id": self.session_id,
                "output_dir": self.output_base_dir
            }


def parse_arguments():
    """Parse command line arguments for the complete pipeline."""
    parser = argparse.ArgumentParser(
        description="BioInsectiNet: Complete Pipeline for Bioinsecticide Design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python main.py --target_fasta "MKVLWAALLVTFLAGCQA..." --run_all
  
  # Run only generation and affinity filtering
  python main.py --target_fasta "MKVL..." --run_generation --run_affinity
  
  # Run GA optimization on existing SMILES file
  python main.py --target_fasta "MKVL..." --input_smiles_file molecules.smi --run_ga
  
  # Run complete pipeline with docking of 5 molecules (GA best + 4 from filtering)
  python main.py --target_fasta "MKVL..." --run_all --num_molecules_docking 5
  
  # Run docking without GA (using top 5 molecules from affinity filtering)
  python main.py --target_fasta "MKVL..." --run_generation --run_affinity --run_docking --num_molecules_docking 5
  
  # Run only docking on existing SMILES file (top 3 molecules)
  python main.py --target_fasta "MKVL..." --input_smiles_file molecules.smi --run_docking --num_molecules_docking 3
  
  # Run complete pipeline with custom parameters
  python main.py --target_fasta "MKVL..." --run_all --total_generated 2000 --generations 50
        """
    )
    
    # Required arguments
    parser.add_argument("--target_fasta", type=str, required=True,
                       help="Target protein sequence in FASTA format")
    
    # Pipeline control arguments
    parser.add_argument("--run_all", action="store_true",
                       help="Run all pipeline steps (generation + affinity + GA + docking)")
    parser.add_argument("--run_generation", action="store_true",
                       help="Run molecule generation step")
    parser.add_argument("--run_affinity", action="store_true",
                       help="Run affinity filtering step")
    parser.add_argument("--run_ga", action="store_true",
                       help="Run genetic algorithm optimization step")
    parser.add_argument("--run_docking", action="store_true",
                       help="Run molecular docking step")
    
    # Input/Output arguments
    parser.add_argument("--input_smiles_file", type=str,
                       help="Input SMILES file (skip generation if provided)")
    parser.add_argument("--output_base_dir", type=str,
                       help="Base output directory (default: bioinsectinet_results_TIMESTAMP)")
    
    # Molecule generation parameters
    parser.add_argument("--generator_model_path", type=str,
                       default="models/generator/bindingDB_smiles_filtered_v1.pth",
                       help="Path to RNN generator model")
    parser.add_argument("--unique_chars_dict_path", type=str,
                       default="models/unique_chars_dict.json",
                       help="Path to unique characters dictionary")
    parser.add_argument("--total_generated", type=int, default=500,
                       help="Total number of molecules to generate")
    parser.add_argument("--min_length", type=int, default=20,
                       help="Minimum SMILES length")
    parser.add_argument("--max_length", type=int, default=150,
                       help="Maximum SMILES length")
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature for generation")
    
    # Affinity filtering parameters
    parser.add_argument("--affinity_model_path", type=str,
                       default="models/checkpoints/cnn_affinity/trial_1_loss_0.1974.pth",
                       help="Path to affinity prediction model")
    parser.add_argument("--db_affinity_path", type=str,
                       default="models/cnn_affinity.db",
                       help="Path to affinity model database")
    parser.add_argument("--study_name", type=str, default="cnn_affinity",
                       help="Study name for hyperparameter optimization")
    parser.add_argument("--accepted_value", type=float, default=1000.0,
                       help="IC50 threshold for affinity filtering")
    parser.add_argument("--max_molecules", type=int, default=50,
                       help="Maximum number of molecules to keep after filtering")
    
    # Genetic algorithm parameters
    parser.add_argument("--objective_ic50", type=float, default=50.0,
                       help="Target IC50 value for genetic algorithm")
    parser.add_argument("--generations", type=int, default=100,
                       help="Number of GA generations")
    parser.add_argument("--population_size", type=int, default=50,
                       help="GA population size")
    parser.add_argument("--num_parents_select", type=int, default=2,
                       help="Number of parents to select for reproduction")
    parser.add_argument("--mutation_rate", type=float, default=0.1,
                       help="Mutation rate for GA")
    parser.add_argument("--stagnation_limit", type=int, default=6,
                       help="Generations without improvement to stop GA")

    
    
    # Docking parameters
    parser.add_argument("--num_molecules_docking", type=int, default=3,
                       help="Number of molecules to use for docking (includes GA best + top N-1 from filtering if GA is used)")
    parser.add_argument("--docking_center", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                       help="Docking box center coordinates (x y z)")
    parser.add_argument("--docking_box_size", type=float, nargs=3, default=[20.0, 20.0, 20.0],
                       help="Docking box size (x y z)")
    parser.add_argument("--vina_path", type=str, default="vina.exe",
                       help="Path to AutoDock Vina executable, if does not exists, it will be downloaded automatically")
    
    # Visualization and output parameters
    parser.add_argument("--draw_lowest", action="store_true",
                       help="Generate images of best molecules")
    parser.add_argument("--smiles_to_draw", type=int, default=3,
                       help="Number of top molecules to visualize")
    parser.add_argument("--upload_to_mega", action="store_true",
                       help="Upload results to Mega cloud storage")
    parser.add_argument("--generate_qr", action="store_true",
                       help="Generate QR codes for result links")
    
    args = parser.parse_args()
    
    if args.run_all:
        args.run_generation = True
        args.run_affinity = True
        args.run_ga = True
        args.run_docking = True
    
    if not any([args.run_generation, args.run_affinity, args.run_ga, args.run_docking]):
        parser.error("You must enable at least one pipeline step (--run_generation, --run_affinity, --run_ga, --run_docking), or use --run_all.")

    elif args.run_generation and args.input_smiles_file:
        parser.error("Cannot use --run_generation with --input_smiles_file. Choose one method for molecule input.")

    elif not args.run_generation and not args.input_smiles_file:
        print("[WARNING] Not generating molecules, neither loading from input file. Using default dataset.")
        args.input_smiles_file = "generated_molecules/generated_molecules_examples.smi"
        filtered_smiles = load_smiles_from_file(args.input_smiles_file)
        print(f"[INFO] Using default dataset with {len(filtered_smiles)} SMILES.")
        if not filtered_smiles:
            parser.error("[ERROR] No SMILES available for processing. Either enable generation or provide input file.")

    return args


def main():
    try:
        args = parse_arguments()
        
        pipeline = BioInsectiNetPipeline(args)

        if args.output_base_dir:
            pipeline.output_base_dir = args.output_base_dir
            os.makedirs(pipeline.output_base_dir, exist_ok=True)
        
        results = pipeline.run_pipeline(args)
        
        if results["success"]:
            print("\n" + "="*60)
            print("\033[92mBIOINSECTINET PIPELINE COMPLETED SUCCESSFULLY!\033[0m")
            print(f"Total execution time: {sum(step.get('time', 0) for step in pipeline.results.values()):.2f} seconds")
            print("="*60)
            print(f"Session ID: {results['session_id']}")
            print(f"Results directory: {results['output_dir']}")
            print(f"Report: {os.path.join(results['output_dir'], 'PIPELINE_REPORT.md')}")
            print("="*60)
            return 0
        else:
            print("\n" + "="*60)
            print("\033[91mBIOINSECTINET PIPELINE FAILED!\033[0m")
            print("="*60)
            print(f"Error: {results['error']}")
            print(f"Logs: {os.path.join(results['output_dir'], 'pipeline_log_*.txt')}")
            print("="*60)
            return 1
            
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
