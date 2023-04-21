import numpy as np
import matplotlib.pyplot as plt

n_inputs = 9
n_outputs = 9
n_hidden_nodes = 9
n_hidden = 2

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

number_of_weights = n_hidden + 1
#how this should be done is we need to create the first layer, repeat for number of layers and last layer

weights_1 = np.random.uniform(low=-2,high=2,size=(n_inputs,n_hidden_nodes))
print(weights_1)

#for n in int(n_hidden-1):
 # new_row = np.random.uniform(low=-2,high=2,size=(n_hidden_nodes,n_hidden_nodes))
  #weights_1 = np.vstack((weights_1,new_row))

#new_row = np.random.uniform(low=-2,high=2,size=(n_hidden_nodes,n_outputs))
#weights_1 = weights_1 = np.vstack((weights_1,new_row))
