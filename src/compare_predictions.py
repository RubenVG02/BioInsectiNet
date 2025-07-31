from check_affinity import predict_affinity
import argparse
import matplotlib.pyplot as plt
import tqdm
import numpy as np
from scripts_with_other_functions.get_hyperparams_db import get_best_trial
import torch
from transformers import AutoTokenizer, AutoModel
from utils.simple_logger import log_info, log_warning, log_error, log_success


def compare_predictions(file_path, model_path, best_trial, total_predictions):
    """
    Compare predictions from a file with the model's predictions.

    Args:
        file_path (str): Path to the file containing SMILES and FASTA sequences.
        model_path (str): Path to the trained model.

    Returns:
        None
    """


    with open(file_path, 'r') as f:
        # skip the header line if it exists
        if f.readline().startswith("SMILES"):
            f.readline()
        lines = f.readlines()

    
    lines = lines[:total_predictions]  

    for line in tqdm.tqdm(lines, desc="Processing lines", total=total_predictions, unit="line"):

        smile = line.strip().split(',')[0]
        fasta = line.strip().split(',')[2] 
        ic_50 = float(line.strip().split(',')[1]) 
        ic_50 = -np.log10(ic_50)  # Convert IC50 to affinity
        predicted_affinity = predict_affinity(smile, fasta, model_path, best_trial=best_trial)

        predicted_values.append(predicted_affinity)
        true_values.append(ic_50)
    

def create_scatter_plot(predicted_values, true_values):
    """
    Create a scatter plot comparing predicted values with true values.

    Args:
        predicted_values (list): List of predicted affinity values.
        true_values (list): List of true IC50 values.

    Returns:
        None
    """

    plt.figure(figsize=(10, 6))
    plt.scatter(true_values, predicted_values, alpha=0.5)
    plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red', linestyle='--')
    plt.xlabel('True IC50 Values')
    plt.ylabel('Predicted Affinity Values')
    plt.title('Comparison of Predicted Affinity vs True IC50 Values')
    plt.grid()
    plt.show()

    if args.plot:
        if args.save_plot:
            plt.savefig(args.save_plot)
        else:
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare predictions from a file with the model's predictions.")
    parser.add_argument("file_path", type=str, help="Path to the file containing SMILES and FASTA sequences.")
    parser.add_argument("model_path", type=str, help="Path to the trained model.")
    parser.add_argument("--db_path", type=str, default="models/cnn_affinity.db", help="Path to the database file containing hyperparameters.")
    parser.add_argument("--plot", action="store_true", help="Whether to create a scatter plot of the predictions.")
    parser.add_argument("--save_plot", type=str, default=None, help="Path to save the scatter plot image. If not provided, the plot will not be saved.")
    parser.add_argument("--total_predictions", type=int, default=10000, help="Total number of predictions to compare. Default is 10000.")

    args = parser.parse_args()
    
    predicted_values = []
    true_values = []

    best_trial = get_best_trial(args.db_path, study_name="cnn_affinity")

    compare_predictions(args.file_path, args.model_path, best_trial, args.total_predictions)
    correlation_coefficient = np.corrcoef(predicted_values, true_values)[0, 1]
    log_info(f"Correlation coefficient between predicted and true values: {correlation_coefficient:.2f}")

    if args.plot:
        create_scatter_plot(predicted_values, true_values)

        
