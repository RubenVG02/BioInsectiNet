
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Input, PReLU, Dropout, concatenate, BatchNormalization
from keras.regularizers import l2



file_path = pd.read_csv(r"your_file_path.csv", sep=",")

# Maximum lengths for SMILES and FASTA sequences
max_smiles = 130
max_fasta = 5000

elements_smiles = ['N', '2', 'H', 'K', 'Z', 'O', 'M', ']', 'c', 'l', '=', '6', ')', 'F', 'o', 'r', '7', 'P','g', '5', 't', '8', '9', '1', '0', 'I', '4', '[', 'i', 'a', 'C', '-', 'n', '#', 'L', '(', 'S', 'B', 'A', 'T', 's', '3', '+', 'e']
elements_fasta = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']  

# elements_smiles refers to the elements by which the smiles can be formed
# If you want to use all of your file, use: file_path.smiles.apply(list).sum(), although it will take a long time in very large file

int_smiles = dict(zip(elements_smiles, range(1, len(elements_smiles)+1)))
int_fasta = dict(zip(elements_fasta, range(1, len(elements_fasta)+1)))

def convert(file_path=file_path):

    '''
    Function to convert all elements (both smiles and fasta) into int, in order to be trained in the model

    Parameters:

        file_path (path): DataFrame containing the SMILES, FASTA and IC50 columns. Columns must be named "smiles", "sequence" and "IC50". This file is generated from src/fix_data_for_models.py

    Returns:

        smiles_w_numbers (list): List of SMILES converted to integers
        fasta_w_numbers (list): List of FASTA converted to integers
    
    '''

    smiles_w_numbers = []
    for i in file_path.smiles:
        smiles_list = [int_smiles.get(element, 0) for element in i]
        smiles_list.extend([0] * (max_smiles - len(smiles_list)))
        smiles_w_numbers.append(smiles_list)

    fasta_w_numbers = []
    for i in file_path.sequence:
        fasta_list = [int_fasta.get(element, 0) for element in i]
        fasta_list.extend([0] * (max_fasta - len(fasta_list)))
        fasta_w_numbers.append(fasta_list)

    ic50_numeros = list(file_path.IC50)

    return smiles_w_numbers, fasta_w_numbers, ic50_numeros



X_test_smile, X_test_fasta, T_test_IC50 = convert(file_path[350000:])


def model_cnn(file_path=file_path):

    '''
    Function to train a model using CNN. The model is trained using the SMILES and FASTA sequences. 
    The model is trained using the IC50 values.

    Parameters:
            file_path (path): DataFrame containing the SMILES, FASTA and IC50 columns. Columns must be named "smiles", "sequence" and "IC50". This file is generated from src/fix_data_for_models.py

    '''
    regulator = l2(0.001)

    # Model for SMILES
    smiles_input = Input(shape=(max_smiles,), dtype='int32', name='smiles_input')
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

    # Model for FASTA
    fasta_input = Input(shape=(max_fasta,), name='fasta_input')
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

    # Concatenate and Dense layers
    combined = concatenate([pool_smiles, pool_fasta])
    dense = Dense(units=1024, activation="relu")(combined)
    dense = Dropout(0.3)(dense)
    dense = Dense(units=1024, activation="relu")(dense)
    dense = Dropout(0.3)(dense)
    dense = Dense(units=512, activation="relu")(dense)

    output = Dense(1, activation="relu", name="output")(dense)

    model = tf.keras.models.Model(inputs=[smiles_input, fasta_input], outputs=[output])

    def r2_score(y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1 - SS_res / (SS_tot + K.epsilon()))

    model.compile(optimizer="adam",
                  loss={'output': "mean_squared_logarithmic_error"},
                  metrics={'output': r2_score})

    save_model_path = "models/cnn_model.hdf5"
    checkpoint = ModelCheckpoint(save_model_path, monitor='val_loss', verbose=1, save_best_only=True)

    size_per_epoch = 50700
    train = file_path[:355000]
    loss = []
    loss_validades = []
    epochs = 50

    for epoch in range(epochs):
        start = 0
        end = size_per_epoch
        print(f"Comenzando el epoch {epoch+1}")

        while end <= 355000:
            X_smiles, X_fasta, y_train = convert(train[start:end])

            r = model.fit({'smiles_input': np.array(X_smiles),
                            'fasta_input': np.array(X_fasta)},
                           {'output': np.array(y_train)},
                           validation_data=({'smiles_input': np.array(X_test_smile),
                                             'fasta_input': np.array(X_test_fasta)},
                                            {'output': np.array(T_test_IC50)}),
                           callbacks=[checkpoint], epochs=1, batch_size=64, shuffle=True)

            start += size_per_epoch
            end += size_per_epoch

        loss.append(np.mean(r.history["loss"]))
        loss_validades.append(np.mean(r.history["val_loss"]))

    plt.plot(range(epochs), loss, label="loss")
    plt.plot(range(epochs), loss_validades, label="val_loss")
    plt.legend()
    plt.show()


# Example usage
model_cnn(file_path=file_path)