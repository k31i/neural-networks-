import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))
#another function needs to be added for the output value as we want the highest probability chosen

data = pd.read_csv("/workspaces/neural-networks-/python code/data/train.csv")
data = np.array(data)
m,n = data.shape
#note the matricies will need to be eduted so we can include the first list of 700 e.c.t

weight_lis = []
for i in range(3):
    matrix = np.random.uniform(low=-2, high=2, size=(10, 10))
    weight_lis.append(matrix)

bais_lis = []
for i in range(3):
    matrix = np.random.uniform(low=-2, high=2, size=(10, 10))
    bais_lis.append(matrix)

node_lis_1 = []
node_lis_2 = np.empty(10)
node_lis_3 = np.empty(10)
node_lis_4 = np.empty(10)
