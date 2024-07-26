import tensorflow as tf
import numpy as np
from keras.layers import CuDNNLSTM, Dropout, Dense, BatchNormalization
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt

data_path = r"path_to_smiles_file.txt" # Path to the file containing the SMILES strings
data = open(data_path).read()

# To get the unique data elements to integers using a dictionary
# Therefore, we associate a numerical value to each letter
elements_smiles = {u: i for i, u in enumerate(sorted(set(data)))}
elements_smiles.update({-1: "\n"})

# To pass from numeric elements to SMILES elements
int_2_elements = {i: u for i, u in enumerate(sorted(set(data)))}
int_2_elements.update({"\n": -1})

map_int = len(elements_smiles)
map_char = len(int_2_elements)


def split_input_target(chunk, values=map_char):
    input_text = chunk[:-1]
    target_idx = chunk[-1]
    target = tf.one_hot(target_idx, depth=values)
    target = tf.reshape(target, [-1])
    return input_text, target


max_smile_length = 137

slices = np.array([[elements_smiles[c]] for c in data])

char_dataset = tf.data.Dataset.from_tensor_slices(slices)

sequences = char_dataset.batch(max_smile_length + 1, drop_remainder=True)

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

# We can modify the model to add more layers or change the number of neurons in each layer
# We can also change the optimizer, the loss function and the metrics
# Depending on the number of different elements in your smile sequence, map_char can be changed, and you can also change it manually depending on your df

weights_path = r"path_to_weights.h5" # Path to the weights of the model
model.load_weights(weights_path)
#This is used to continue training a model that has already been trained

model.compile(optimizer="adam",
               loss="categorical_crossentropy", metrics=["accuracy"])
# Different loss functions can be used, but I reccomend categorical_crossentropy

filepath = r"path_to_save_model.h5" #Path to save the model
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True, mode='min')
callbacks_list = [checkpoint]

r = model.fit(dataset, epochs=150, callbacks=callbacks_list, batch_size=128)

# We can plot the loss and accuracy of the model

plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()