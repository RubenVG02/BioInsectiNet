from check_affinity import calculate_affinity
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def create_graph(path_data_csv):
    """
    Create a scatter plot comparing real values and predicted values.

    Parameters:
        path_data_csv (str): Path to the CSV file containing the data.
    """
    # Load the data from the CSV file
    real_data = pd.read_csv(path_data_csv, sep=",", header=0, names=["smiles", "sequence", "IC50"])

    # Extract columns from the DataFrame
    smiles = np.array(real_data["smiles"])
    fasta_sequences = np.array(real_data["sequence"], dtype="S")
    ic50_values = np.array(real_data["IC50"], dtype="f")

    # List to store predictions
    predictions = []

    # Generate predictions
    for i in range(min(50, len(fasta_sequences))):
        prediction = calculate_affinity(smile="CCN(CCO)CC(=O)N1CC[C@@H](C(=O)N[C@H]2C[C@@H](C)O[C@@H](C)C2)CC1", fasta=fasta_sequences[i])
        predictions.append(prediction)

    # Truncate ic50_values to match the number of predictions
    ic50_values_truncated = ic50_values[:len(predictions)]

    # Create the scatter plot
    plt.scatter(predictions, ic50_values_truncated)
    plt.xlabel("Predictions")
    plt.ylabel("Real values")
    plt.title("Predicted vs Real IC50 Values")
    plt.show()

# Example usage
path_to_csv = "path_to_your_data.csv"  # Update this with the actual path to your CSV file
create_graph(path_to_csv)
