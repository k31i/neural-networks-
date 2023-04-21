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

