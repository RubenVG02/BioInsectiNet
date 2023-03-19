import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, PReLU, Dropout, concatenate, BatchNormalization
from keras.regularizers import l2



def calculate_affinity(path_csv=r"C:\Users\ASUS\Desktop\github22\dasdsd\CSV\500k_dades.csv", smile="", fasta="", path_model=r"cnn_model.hdf5"):
    #path = pd.read_csv(f"{path_csv}", sep=",")

    # maximum value that I want my smileys to have, they will be used to train the model
    max_smiles = 100
    elements_smiles = ['6', '3', '=', 'H', 'C', 'O', 'c', '#', 'a', '[', 't', 'r', 'K', 'n', 'B', 'F', '4', '+', ']', '-', '1', 'P',
                       '0', 'L', 'g', '9', 'Z', '(', 'N', '8', 'I', '7', '5', 'l', ')', 'A', 'e', 'o', 'V', 's', 'S', '2', 'M', 'T', 'u', 'i', "p"]
    # elements_smiles refers to the elements by which the smiles can be formed

    int_smiles = dict(zip(elements_smiles, range(1, len(elements_smiles)+1)))
    # To associate all elements with a given int

    max_fasta = 5000
    elements_fasta = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
                      'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']  # Formed by the different aa that make up the fasta

    int_fasta = dict(zip(elements_fasta, range(1, len(elements_fasta)+1)))
    #To associate all elements with a given int (range is 1, len+1 because they are filled with zeros to reach maximum_fasta)
    
    #kernel regulator
    regulator = l2(0.001)


    smiles_input = tf.keras.Input(
        shape=(max_smiles,), dtype='int32', name='smiles_input')
    embed = Embedding(input_dim=len(
        elements_smiles)+1, input_length=max_smiles, output_dim=128)(smiles_input)
    x = Conv1D(
        filters=32, kernel_size=3, padding="SAME", input_shape=(4000, max_smiles))(embed)
    x = PReLU()(x)

    x = Conv1D(filters=64, kernel_size=3, padding="SAME")(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv1D(
        filters=128, kernel_size=3, padding="SAME")(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    pool = GlobalMaxPooling1D()(
        x)  # maxpool per obtenir un vector de 1d

    # model per fastas
    fasta_input = tf.keras.Input(shape=(max_fasta,), name='fasta_input')
    embed2 = Embedding(input_dim=len(
        elements_fasta)+1, input_length=max_fasta, output_dim=256)(fasta_input)
    x2 = Conv1D(
        filters=32, kernel_size=3, padding="SAME", input_shape=(4000, max_fasta))(embed2)
    x2 = PReLU()(embed2)

    x2 = Conv1D(
        filters=64, kernel_size=3, padding="SAME")(x2)
    x2 = BatchNormalization()(x2)
    x2 = PReLU()(x2)
    x2 = Conv1D(
        filters=128, kernel_size=3, padding="SAME")(x2)
    x2 = BatchNormalization()(x2)
    x2 = PReLU()(x2)
    pool2 = GlobalMaxPooling1D()(
        x2)  #maxpool to get a 1d vector

    junt = concatenate(inputs=[pool, pool2])

    # dense

    de = Dense(units=1024, activation="relu")(junt)
    dr = Dropout(0.3)(de)
    de = Dense(units=1024, activation="relu")(dr)
    dr = Dropout(0.3)(de)
    de2 = Dense(units=512, activation="relu")(dr)

    # output

    output = Dense(
        1, activation="relu", name="output", kernel_initializer="normal")(de2)

    modelo = tf.keras.models.Model(
        inputs=[smiles_input, fasta_input], outputs=[output])

    modelo.load_weights(f"{path_model}")
    smiles_in = []
    for element in smile:
        smiles_in.append(int_smiles[element])
    while (len(smiles_in) != max_smiles):
        smiles_in.append(0)

    fasta_in = []
    for amino in fasta:
        fasta_in.append(int_fasta[amino])
    while (len(fasta_in) != max_fasta):
        fasta_in.append(0)

    predict = modelo.predict({'smiles_input': np.array(smiles_in).reshape(1, 100,),
                               'fasta_input': np.array(fasta_in).reshape(1, 5000,)})[0][0]
    print(predict)

    return predict


# mesurador_afinitat(smile="CSc1ccccc1-c1ccccc1-c1nnnn1-c1ccccc1F",fasta = "MGGDLVLGLGALRRRKRLLEQEKSLAGWALVLAGTGIGLMVLHAEMLWFGGCSWALYLFLVKCTISISTFLLLCLIVAFHAKEVQLFMTDNGLRDWRVALTGRQAAQIVLELVVCGLHPAPVRGPPCVQDLGAPLTSPQPWPGFLGQGEALLSLAMLLRLYLVPRAVLLRSGVLLNASYRSIGALNQVRFRHWFVAKLYMNTHPGRLLLGLTLGLWLTTAWVLSVAERQAVNATGHLSDTLWLIPITFLTIGYGDVVPGTMWGKIVCLCTGVMGVCCTALLVAVVARKLEFNKAEKHVHNFMMDIQYTKEMKESAARVLQEAWMFYKHTRRKESHAARRHQRKLLAAINAFRQVRLKHRKLREQVNSMVDISKMHMILYDLQQNLSSSHRALEKQIDTLAGKLDALTELLSTALGPRQLPEPSQQSK")
