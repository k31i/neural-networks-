import numpy as np
import matplotlib.pyplot as plt

n_inputs = 9
n_outputs = 9
n_hidden_nodes = 9
n_hidden = 2

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

number_of_weights = n_hidden -1
#how this should be done is we need to create the first layer, repeat for number of layers and last layer

#weights list
weight_lis = []
weight_lis.append(np.random.uniform(low=-2,high=2,size=(n_inputs,n_hidden_nodes)))

for n in range(number_of_weights):
  new_row = np.random.uniform(low=-2,high=2,size=(n_hidden_nodes,n_hidden_nodes))
  weight_lis.append(new_row)

new_row = np.random.uniform(low=-2,high=2,size=(n_hidden_nodes,n_outputs))
weight_lis.append(new_row)
#appending will allow us to create a list of matricies

#bais list
bais_lis = []
bais_lis.append(np.random.uniform(low=-2,high=2,size=(n_inputs,n_hidden_nodes)))

for n in range(number_of_weights):
  new_row = np.random.uniform(low=-2,high=2,size=(n_hidden_nodes,n_hidden_nodes))
  bais_lis.append(new_row)

new_row = np.random.uniform(low=-2,high=2,size=(n_hidden_nodes,n_outputs))
bais_lis.append(new_row)
#appending will allow us to create a list of matricies