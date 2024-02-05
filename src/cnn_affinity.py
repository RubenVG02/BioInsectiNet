
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D, Embedding, Input, PReLU, Dropout, concatenate, BatchNormalization
from keras.regularizers import l2



arx = pd.read_csv(r"", sep=",")

# maximum value that I want my smiles to have, they will be used to train the model
max_smiles = 130
elements_smiles = ['N', '2', 'H', 'K', 'Z', 'O', 'M', ']', 'c', 'l', '=', '6', ')', 'F', 'o', 'r', '7', 'P','g', '5', 't', '8', '9', '1', '0', 'I', '4', '[', 'i', 'a', 'C', '-', 'n', '#', 'L', '(', 'S', 'B', 'A', 'T', 's', '3', '+', 'e']
# elements_smiles refers to the elements by which the smiles can be formed
# if you want to use all of your arx: arx.smiles.apply(list).sum(), although it will take a long time in very large arx

int_smiles = dict(zip(elements_smiles, range(1, len(elements_smiles)+1)))
# To associate all elements with a given int (range is 1, len+1 because they are filled with zeros to reach maximum_smiles)

max_fasta = 5000
elements_fasta = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K',
                  'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']  # Formed by the different aa that make up the fasta

int_fasta = dict(zip(elements_fasta, range(1, len(elements_fasta))))
# To associate all elements with a given int (range is 1, len+1 because they are filled with zeros to reach maximum_fasta)


def convert(arx=arx):

    #Function to convert all elements (both smiles and fasta) into int, in order to be trained in the model

    smiles_w_numbers = []  # Smiles obtained with int_smiles[1] and the smiles of the df
    for i in arx.smiles:
        smiles_list = []
        for elements in i:  # Elements refers to the elements that make up elements_smile
            try:
                smiles_list.append(int_smiles[elements])
            except:
                pass
        while (len(smiles_list) != max_smiles):
            smiles_list.append(0)
        smiles_w_numbers.append(smiles_list)

    fasta_w_numbers = []
    for i in arx.sequence:
        fasta_list = []
        for elements in i:  # Elements fa referència a els elements que formen elements_smile
            try:
                fasta_list.append(int_fasta[elements])
            except:
                pass
        while (len(fasta_list) != max_fasta):
            fasta_list.append(0)
        fasta_w_numbers.append(fasta_list)

    ic50_numeros = list(arx.IC50)

    return smiles_w_numbers, fasta_w_numbers, ic50_numeros


X_test_smile, X_test_fasta, T_test_IC50 = convert(arx[350000:])


def model_cnn():
    # model to train 
    
    # kernel regularizer
    regulatos = l2(0.001)

    # model per a smiles
    smiles_input = Input(
        shape=(max_smiles,), dtype='int32', name='smiles_input')
    embed = Embedding(input_dim=len(
        elements_smiles)+1, input_length=max_smiles, output_dim=128)(smiles_input)
    x = Conv1D(
        filters=32, kernel_size=3, padding="SAME", input_shape=(50700, max_smiles))(embed)
    x = PReLU()(x)

    x = Conv1D(filters=64, kernel_size=3, padding="SAME")(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    x = Conv1D(
        filters=128, kernel_size=3, padding="SAME")(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    pool = GlobalMaxPooling1D()(
        x)  # maxpool to get a 1d vector

    # model per fastas
    fasta_input = Input(shape=(max_fasta,), name='fasta_input')
    embed2 = Embedding(input_dim=len(
        elements_fasta)+1, input_length=max_fasta, output_dim=256)(fasta_input)
    x2 = Conv1D(
        filters=32, kernel_size=3, padding="SAME", input_shape=(50700, max_fasta))(embed2)
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

    model = tf.keras.models.Model(
        inputs=[smiles_input, fasta_input], outputs=[output])

 
    # funció per mirar la precisió del model (serà la nostra metric)
    def r2_score(y_true, y_pred):
        SS_res = K.sum(K.square(y_true - y_pred))
        SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
        return (1-SS_res/(SS_tot)+K.epsilon())

    model.load_weights(
        r"")
    # In case you want to continue training a model
    
    model.compile(optimizer="adam",
                   loss={'output': "mean_squared_logarithmic_error"},
                   metrics={'output': r2_score})
    
    # To do checkpoints
    save_model_path = "models/cnn_model.hdf5"
    checkpoint = ModelCheckpoint(save_model_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True)

    # We use a high value to get better results
    size_per_epoch = 50700

    train = arx[:355000]
    loss = []
    loss_validades = []
    epochs = 50

    for epoch in range(epochs):  #Amount of epochs you want to use
        start = 0
        end = size_per_epoch
        print(f"Començant el epoch {epoch+1}")

        while final < 355000:
            X_smiles, X_fasta, y_train = convert(train[start:end])

            r = model.fit({'smiles_input': np.array(X_smiles),
                            'fasta_input': np.array(X_fasta)}, {'output': np.array(y_train)},
                           validation_data=({'smiles_input': np.array(X_test_smile),
                                             'fasta_input': np.array(X_test_fasta)}, {'output': np.array(T_test_IC50)}),  callbacks=[checkpoint], epochs=20, batch_size=64, shuffle=True)

            inici += size_per_epoch
            final += size_per_epoch

        loss.append(r.history["loss"])
        loss_validades.append(r.history["val_loss"])

    plt.plot(range(epochs), loss, label="loss")
    plt.plot(range(epochs), loss_validades, label="val_loss")
    plt.legend()
    plt.show()


model_cnn()
