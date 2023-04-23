import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#so we will have a simple 2 hiden layer system with the hiden layers having 16 nodes.
# the inital values will be 784 nodes and the output will be 10

#-------opening the mnist file and organising it

data = pd.read_csv("/workspaces/neural-networks-/python code/data/train.csv")
data = np.array(data)
m, n = data.shape
np.random.shuffle(data) # shuffle before splitting into dev and training sets

data_dev = data[0:1000].T
Y_dev = data_dev[0]
X_dev = data_dev[1:n]
X_dev = X_dev / 255.

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape


#-------define the pramaters aka weights and nodes

inp_a = []
nodes_1_a = np.empty(16)
nodes_1_z = np.empty(16)
nodes_2_a = np.empty(16)
nodes_2_b = np.empty(16)
out_a = np.empty(10)
out_z = np.empty(10)

weig_1 = np.random.uniform(low=-2, high=2, size=(784, 16))
weig_2 = np.random.uniform(low=-2, high=2, size=(16, 16))
weig_3 = np.random.uniform(low=-2, high=2, size=(16, 10))

bias_1 = np.random.uniform(low=-1,high=1,size=16)
bias_2 = np.random.uniform(low=-1,high=1,size=16)
bias_3 = np.random.uniform(low=-1,high=1,size=10)

#-------define the functions to be used. so forwarad and back propagation


def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))
#another function needs to be added for the output value as we want the highest probability chosen
def softactive(x):
   return np.exp(x) / sum(np.exp(x))