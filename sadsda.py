import random
import numpy as np
import rdkit.Chem as Chem

def childs(*parents):
    '''
    Function to cross two smile sequences in order to obtain two new molecules. Function used when ic50 value does not improve during the generations.
    
    '''
    parents1=[]
    for i in parents:
        parents1.append(i)
    if len(parents1[1])>len(parents1[0]):
        crossover_point=random.randint(0, len(parents[0])-1)
    else:
        crossover_point=random.randint(0, len(parents[1])-1)
    child1 = parents[0][:crossover_point] + parents[1][crossover_point:]
    child2 = parents[1][:crossover_point] + parents[0][crossover_point:]
    print(child1, child2)
    child1= Chem.MolFromSmiles(child1)
    child2= Chem.MolFromSmiles(child2)
    if child1 is not None and child2 is not None:
        return child1, child2
    else:
        childs() 

lista=["B=C(CCC)Cp1cnnc1CBC(=B)c1ccc(I)cc1N", "B=C(CCC)Cp1cnnc1CBC(=B)c1ccc(I)cc1N"]
childs(lista[0], lista[1])