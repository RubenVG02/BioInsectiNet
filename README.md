# Fungic Bioinsecticide Discovery


## Usage

First of all, to make a good prediction, the target proteins must be in the FASTA (amino acids) sequence.

Then, you need 2 neural networks, one for toxicity prediction and another to generate bioinsecticides. You can train them yourself using `cnn_affinity.py` for toxicity prediction and `generate_rnn.py` for bioinsecticide generation. You can also use the ones I have trained, which are in the `definitive_models` folder.

To train your models you need data, which can be obtained from different databases such as Chembl, Pubchem, etc. You can also use the one I have used, which is "insect.csv".

### CNN USAGE (Affinity) ###
To use the CNN, use `check_affinity.py`. You need to specify the path to the model, the path to the data, and the path to the target protein. The program will return the toxicity of the designed bioinsecticides using the `calculate_affinity` function.

### RNN USAGE (Generation) ###

To use the RNN, use `pretrained_rnn.py`. You need to specify the path to the model, the path to the data, and the path to the target protein. The program will return the designed bioinsecticides using the generate function.

### COMBINATION ###

To use the combination of both models, use `affinity_with_target_and_generator.py`. You need to specify the path to the model, the path to the data, and the path to the target protein. The program will return the designed bioinsecticides using the generate function. You can also specify the toxicity limit of the designed bioinsecticides using the calculate_affinity function. The program will return the designed bioinsecticides with a lower toxicity than the limit. You can also specify a path of generated molecules to check.

### GENETIC ALGORITHM ###

To use the genetic algorithm, use `genetic_algorithm.py`. You can use 3 paths:
- Smile sequences using lists [smile1, smile2, smile3, ...]
- Smile sequences using a CSV file
- RNN model

Then, you need to specify your model path, the number of generations, and the destination path. The program will return the best smile sequence of the last generation.

### 3D STRUCTURE ###

To obtain the 3D structure of the designed bioinsecticides, use `3d_repr.py`. You need to specify the path to the model, the path to the data, and the path to the target protein. The program will return the designed bioinsecticides using the generate function. You can also specify the toxicity limit of the designed bioinsecticides using the calculate_affinity function. The program will return the designed bioinsecticides with a lower toxicity than the limit. You can also specify a path of generated molecules to check. You will obtain an sdf file with the 3D structure of the designed bioinsecticides.
Then, using PyMOL, you can obtain the 3D structure of the designed bioinsecticides in different formats (sdf, pdb, etc.) by using the "pymol_3d.py" script directly in PyMOL.



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
- Obtaining CSV files and screenshots of the results
- Obtaining the 3D structure of the designed bioinsecticides in different formats (sdf, pdb, etc.)
- Fast and easy to use


## Future Improvements

- Add more databases to the CNN
- Add more databases to the RNN
- More complexity to the GA
- Directly obtain the 3D structure of the designed bioinsecticides


## License

[MIT](https://choosealicense.com/licenses/mit/)

## Acknowledgements

- [Keras](https://keras.io/)
- [Tensorflow](https://www.tensorflow.org/)
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)




