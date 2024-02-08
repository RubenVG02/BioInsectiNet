import tensorflow as tf
import tensorflow_datasets as tfds

from keras.layers import CuDNNLSTM
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import BatchNormalization

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import Draw

import numpy as np
import sys
import random
import time

from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

colorama_init()



def generator(path_model=r"models\definitive_models\rnn_model.hdf5", path_data=r"samples/txt_files/98k.txt",
              number_generated=100, img_druglike=True, path_destination_molecules=r""):
    '''
        Parameters:
        -path_model: Path where the already trained model is located
        -path_data: Path where the data used is located (path model and path data must have the same dimensions/same amount of different elements)
        -number_generated: Number of molecules that are generated, by default 100
        -img_druglike: To generate .jpg images of the generated drug-like molecules (they will be ordered according to epoch) (By default True)
        -Path_destination_molecules: Destination path of the generated SMILE sequences
    '''
    def split_input_target(valors):
        input_text = valors[:-1]
        target_idx = valors[-1]
        target = tf.one_hot(target_idx, depth=map_char)  #depth must be equal to the number of different outputs of the pre-trained model
        target = tf.reshape(target, [-1])
        return input_text, target
    
    def create_seed(max_molecules=137):
        '''
        Function that takes your ds and allows you to obtain a seed from it to generate molecules
        
        Parameters:
            -max_molecules: Maximum length you want your pattern/seed to have, by default, 137
        
        '''
        generador_seeds=tfds.as_numpy(dataset.take(random.randint(0, len(dades))).take(1))
        for a, b in enumerate(generador_seeds):
            break
        pattern=b[0][np.random.randint(0,max_molecules)]
        return pattern

    with open(f"{path_data}") as f:
        dades = "\n".join(line.strip() for line in f)


    elements_smiles = {u: i for i, u in enumerate(sorted(set(dades)))}
    elements_smiles.update({-1: "\n"})


    int_2_elements = {i: c for i, c in enumerate(elements_smiles)}
    int_2_elements.update({"\n": -1})

    map_int = len(elements_smiles)
    map_char = len(int_2_elements)
    
    

    slices = np.array([[elements_smiles[c]] for c in dades])

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(slices)

    sequences = char_dataset.batch(137+1, drop_remainder=True)

    dataset = sequences.map(split_input_target)

    dataset = dataset.shuffle(20000).batch(256, drop_remainder=True)
    

    

    def create_model():
        model = tf.keras.models.Sequential([CuDNNLSTM(128, input_shape=(137, 1), return_sequences=True),
                                            Dropout(0.1),
                                            CuDNNLSTM(
                                                256, return_sequences=True),
                                            BatchNormalization(),
                                            Dropout(0.1),
                                            CuDNNLSTM(
                                                512, return_sequences=True),
                                            BatchNormalization(),
                                            Dropout(0.1),
                                            CuDNNLSTM(
                                                256, return_sequences=True),
                                            BatchNormalization(),
                                            Dropout(0.1),
                                            CuDNNLSTM(128),
                                            Dropout(0.1),
                                            Dense(map_char-1, activation="softmax")])
        return model

    model = create_model()
    model.load_weights(f"{path_model}")
    model.compile(loss='categorical_crossentropy', optimizer='adam')

###GENERATION OF MOLECULES###
    seq_length = 137
    pattern=create_seed(max_molecules=seq_length)
    print("\"", ''.join([int_2_elements[value[0]] for value in pattern]), "\"")
    final = ""
    total_smiles = []
    for i in range(number_generated):
        for i in range(random.randrange(40,137)):
            x = np.reshape(pattern, (1, len(pattern)))
            prediction = model.predict(x, verbose=0)
            index = np.argmax(prediction)  #Get the maximum value from the prediction array
            result = int_2_elements[index]
            print(result, end="" )
            final += result
            pattern=np.append(pattern, index)
            pattern = pattern[1:len(pattern)]
        final = final.split("\n")
        if i%10==0:
            print("\n\n\nChanging seed...\n\n\n")
            pattern=create_seed(max_molecules=seq_length)  #Change the seed in order to obtain better results and more diversity
            print("\"", ''.join([int_2_elements[value[0]] for value in pattern]), "\"")
        for i in final:
            mol1 = Chem.MolFromSmiles(i)
            if len(i) > 20:
                if mol1 == None:
                    print("error")
                elif not mol1 == None:
                    print(result)
                    print(f"{Fore.YELLOW}A chemically possible molecule has come out, I'll check if it's drug-like{Style.RESET_ALL}")
                    if Descriptors.ExactMolWt(mol1) < 500 and Descriptors.MolLogP(mol1) < 5 and Descriptors.NumHDonors(mol1) < 5 and Descriptors.NumHAcceptors(mol1) < 10:
                        #All the conditions that a molecule must meet to be considered drug-like  
                        with open(f"{path_destination_molecules}", "a") as file:
                            with open(f"{path_destination_molecules}", "r") as f:
                                linies = [linea.rstrip() for linea in f]
                            if f"{i}" not in linies:
                                file.write(i + "\n")
                                if img_druglike == True:
                                    Draw.MolToImageFile(
                                mol1, filename=fr"examples/generated_molecules/molecula{int(time.time())}.jpg", size=(400, 300))
                        total_smiles.append(i)
                        print(f"{Fore.GREEN}The obtained molecule is drug-like{Style.RESET_ALL}")
            else:
                pass
        final = ""
    return total_smiles

#Example of use
generator(path_model=r"models\specific_RNN_models\modelo_rnn_insectos.hdf5", path_data=r"samples\txt_files\insectos.txt",
              number_generated=100, img_druglike=True, path_destination_molecules=r"examples/generated_molecules/generated_molecules.txt")
