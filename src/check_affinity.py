import pandas as pd
import tensorflow as tf
import numpy as np
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, PReLU, Dropout, concatenate, BatchNormalization
from keras.regularizers import l2

def calculate_affinity(smile, fasta, path_model):
    """
    Calculates the affinity score for a given SMILES and FASTA sequence using a trained model.

    Parameters:
        smile (str): SMILES string representing the molecule.
        fasta (str): FASTA sequence representing the target protein.
        path_model (path): Path to the trained model file (.h5).

    Returns:
        float: The affinity score predicted by the model.
    """

    # Maximum value that I want my SMILES to have, they will be used to train the model
    max_smiles = 100
    max_fasta = 5000

    # Define elements for SMILES and FASTA encoding
    elements_smiles = ['6', '3', '=', 'H', 'C', 'O', 'c', '#', 'a', '[', 't', 'r', 'K', 'n', 'B', 'F', '4', '+', ']', '-', '1', 'P',
                       '0', 'L', 'g', '9', 'Z', '(', 'N', '8', 'I', '7', '5', 'l', ')', 'A', 'e', 'o', 'V', 's', 'S', '2', 'M', 'T', 'u', 'i', "p"]
    elements_fasta = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
                      'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

    int_smiles = dict(zip(elements_smiles, range(1, len(elements_smiles)+1)))
    int_fasta = dict(zip(elements_fasta, range(1, len(elements_fasta)+1)))

    # Kernel regularizer, not necessary but it can help to avoid overfitting
    regulator = l2(0.001)

    smiles_input = tf.keras.Input(shape=(max_smiles,), dtype='int32', name='smiles_input')
    embed_smiles = Embedding(input_dim=len(elements_smiles)+1, input_length=max_smiles, output_dim=128)(smiles_input)
    x = Conv1D(filters=32, kernel_size=3, padding="SAME", kernel_regularizer=regulator)(embed_smiles)
    x = PReLU()(x)

    x = Conv1D(filters=64, kernel_size=3, padding="SAME")(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv1D(filters=128, kernel_size=3, padding="SAME")(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    pool_smiles = GlobalMaxPooling1D()(x)

    fasta_input = tf.keras.Input(shape=(max_fasta,), name='fasta_input')
    embed_fasta = Embedding(input_dim=len(elements_fasta)+1, input_length=max_fasta, output_dim=256)(fasta_input)
    x2 = Conv1D(filters=32, kernel_size=3, padding="SAME")(embed_fasta)
    x2 = PReLU()(x2)

    x2 = Conv1D(filters=64, kernel_size=3, padding="SAME")(x2)
    x2 = BatchNormalization()(x2)
    x2 = PReLU()(x2)
    x2 = Conv1D(filters=128, kernel_size=3, padding="SAME")(x2)
    x2 = BatchNormalization()(x2)
    x2 = PReLU()(x2)
    pool_fasta = GlobalMaxPooling1D()(x2)

    combined = concatenate([pool_smiles, pool_fasta])

    dense = Dense(units=1024, activation="relu")(combined)
    dense = Dropout(0.3)(dense)
    dense = Dense(units=1024, activation="relu")(dense)
    dense = Dropout(0.3)(dense)
    dense = Dense(units=512, activation="relu")(dense)

    output = Dense(1, activation="relu", name="output")(dense)

    model = tf.keras.models.Model(inputs=[smiles_input, fasta_input], outputs=[output])

    # Load the trained model weights
    model.load_weights(path_model)

    # Prepare the input data
    smiles_in = [int_smiles.get(element, 0) for element in smile]
    smiles_in = smiles_in + [0] * (max_smiles - len(smiles_in))
    
    fasta_in = [int_fasta.get(amino, 0) for amino in fasta]
    fasta_in = fasta_in + [0] * (max_fasta - len(fasta_in))

    # Predict the affinity
    predict = model.predict({
        'smiles_input': np.array(smiles_in).reshape(1, max_smiles),
        'fasta_input': np.array(fasta_in).reshape(1, max_fasta)
    })[0][0]

    print(f"Predicted affinity: {predict}")

    return predict
