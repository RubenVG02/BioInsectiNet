import tensorflow as tf
import pandas as pd
from rdkit import DataStructs, Chem
import numpy as np
from keras.layers import CuDNNLSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import BatchNormalization
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


data = open(r"samples\txt_files\98k.txt").read()

# to get the unique data elements to integers using a dictionary
# so we associate a numerical value to each letter
elements_smiles = {u: i for i, u in enumerate(sorted(set(data)))}
elements_smiles.update({-1: "\n"})

# to pass the numeric elements to smile elements
int_2_elements = {i: u for i, u in enumerate(sorted(set(data)))}
int_2_elements.update({"\n": -1})

map_int = len(elements_smiles)
map_char = len(int_2_elements)


def split_input_target(chunk, values=map_char):
    # Function to split the input and the target
    input_text = chunk[:-1]
    target_idx = chunk[-1]
    target = tf.one_hot(target_idx, depth=values)
    target = tf.reshape(target, [-1])
    return input_text, target


max_smile = 137

slices = np.array([[elements_smiles[c]] for c in data])


char_dataset = tf.data.Dataset.from_tensor_slices(slices)

sequences = char_dataset.batch(137+1, drop_remainder=True)

dataset = sequences.map(split_input_target)

dataset = dataset.shuffle(20000).batch(128, drop_remainder=True)

model = tf.keras.models.Sequential([CuDNNLSTM(128, input_shape=(137, 1), return_sequences=True),
                                     Dropout(0.15),
                                     CuDNNLSTM(256, return_sequences=True),
                                     BatchNormalization(),
                                     Dropout(0.15),
                                     CuDNNLSTM(512, return_sequences=True),
                                     BatchNormalization(),
                                     Dropout(0.15),
                                     CuDNNLSTM(256, return_sequences=True),
                                     BatchNormalization(),
                                     Dropout(0.15),
                                     CuDNNLSTM(128),
                                     Dropout(0.15),
                                     Dense(map_char, activation="softmax")])

#We can modify the model to add more layers or change the number of neurons in each layer
#We can also change the optimizer, the loss function and the metrics
#Depending on the number of different elements in your smile sequence, map_char can be changed, and you can also change it manually depending on your df


model.load_weights(
    r"models\definitive_models\rnn_model.hdf5")
#This is used to continue training a model that has already been trained
model.compile(optimizer="adam",
               loss="categorical_crossentropy", metrics=["accuracy"])
#Different loss functions can be used, but I reccomend categorical_crossentropy

filepath = "" #Path to save the model
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True, mode='min')
callbacks_list = [checkpoint]

r = model.fit(dataset, epochs=100, callbacks=callbacks_list, batch_size=128)

plt.plot(r.history["accuracy"], label="accuracy")
plt.legend()
plt.show()
plt.savefig("acc.png")
