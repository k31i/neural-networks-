import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))
#another function needs to be added for the output value as we want the highest probability chosen
def softactive(x):
   return 0



data = pd.read_csv("/workspaces/neural-networks-/python code/data/train.csv")
data = np.array(data)
m,n = data.shape
#note the matricies will need to be eduted so we can include the first list of 700 e.c.t

#-------------coppied from the YT channel on kaggel
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
#-------------coppied from the YT channel on kaggel

weight_lis = []
matrix = np.random.uniform(low=-2, high=2, size=(784, 10))
weight_lis.append(matrix)

for i in range(2):
    matrix = np.random.uniform(low=-2, high=2, size=(10, 10))
    weight_lis.append(matrix)

bais_lis = []
for i in range(3):
    matrix = np.random.uniform(low=-2, high=2, size=(1, 10))
    weight_lis.append(matrix)


node_lis_1 = []
node_lis_2 = np.empty(10)#hidden layer
node_lis_3 = np.empty(10)#hidden layer
node_lis_4 = np.empty(10)#output

def forwardprop (i,w,b):
   return (np.dot(i,w)) + b

def fix(w,c_dev):
   return w - 0.1*(c_dev)


#c total caculated
num = 0
c = 0
for i in range(10):
   if num == i:
    c = c + (node_lis_4[i] - 1)**2
else:
   c = c + (node_lis_4[i])**2