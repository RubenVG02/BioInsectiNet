# Fungic Bioinsecticide Discovery


## Usage

First of all, in order to make a good prediction, the target proteins must be in FASTA (aminoacids) sequence.

Then, you need 2 neural networks, one for the toxicity prediction and another one to generate bioinsecticides. You can train them youself using "cnn_affinity.py" for the toxicity prediction and "generate_rnn.py" for the bioinsecticide generation. You can also use the ones I have trained, which are in the "definitive_models" folder.

To train your models you need data, which can be obtained from different databases such as Chembl, Pubchem, etc. You can also use the ones I have used, which is "insect.csv".

### CNN USAGE (Affinity) ###
In order to use the CNN, use "check_affinity.py". You need to specify the path to the model, the path to the data and the path to the target protein. The program will return the toxicity of the designed bioinsecticides using the calculate_affinity function.

### RNN USAGE (Generation) ###

In order to use the RNN, use "pretrained_rnn.py". You need to specify the path to the model, the path to the data and the path to the target protein. The program will return the designed bioinsecticides using the generate function.

### COMBINATION ###

In order to use the combination of both models, use "affinity_with_target_and_generator.py". You need to specify the path to the model, the path to the data and the path to the target protein. The program will return the designed bioinsecticides using the generate function. You can also specify the toxicity limit of the designed bioinsecticides using the calculate_affinity function. The program will return the designed bioinsecticides that have a toxicity lower than the limit. You can also specify a path of generated molecules to check.

### GENETIC ALGORITHM ###

In order to use the genetic algorithm, use "genetic_algorithm.py". You can use 3 paths:
- Smile sequences using lists [smile1, smile2, smile3, ...]
- Smile sequences using a csv file
- RNN model

Then, you need to specify your model path, the number of generations, and destination path. The program will return the best smile sequence of the last generation.



## Installation

To use this project, you need to have Python 3.7 or higher installed. Then, you need to install the following libraries:
- Keras
- Tensorflow
- Numpy
- Pandas
- Matplotlib




## Authors

- [@RubenVG02](https://www.github.com/RubenVG02)


## Features

- Design of new bioinsecticides based on the target protein
- Improving the structure of previously designed bioinsecticides based on the target protein
- Predicting the toxicity of the designed bioinsecticides
- Obtaining csv files and screenshots of the results
- Obtaining the 3D structure of the designed bioinsecticides
- Fast and easy to use


