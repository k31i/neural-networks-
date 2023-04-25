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

def one_hot(Y):#################################coppied from kaggel and used in back prop
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def forward_prop(a_0,a_1,a_2,a_3,z_1,z_2,z_3,w_1,w_2,w_3,b_1,b_2,b_3):#correct for dot prod
   z_1 = w_1.dot(a_0) - b_1
   z_2 = w_2.dot(a_1) - b_2
   z_3 = w_3.dot(a_2) - b_3 
   
   a_1 = sigmoid(z_1)
   a_2 = sigmoid(z_2)
   a_3 = softactive(z_3)
   return z_1,z_2,z_3,a_1,a_2,a_3

def sigmoid_dev(x):
   return (1/(1+np.e**(-x)))*(1-(1/(1+np.e**(-x))))

#we need one for soft active derivative we will call it softactive_dev!!!!!!!!

def back_prop(Y,a_3,a_2,a_1,a_0,z_3,z_2,z_1,w_3,w_2):
   si_3 = np.dot(2*(a_3-Y),softactive_dev(z_3))
   b_3_dev = si_3
   w_3_dev = np.dot(a_2,si_3)
   si_2 = np.dot(np.dot(np.transpose(w_3),si_3),sigmoid_dev(z_2))
   b_2_dev = si_2
   w_2_dev = np.dot(a_1,si_2)
   si_1 = np.dot(np.dot(np.transpose(w_2),si_2),sigmoid_dev(z_1))
   b_1_dev = si_1
   w_1_dev = np.dot(a_0,si_1)
   return b_1_dev,b_2_dev,b_1_dev,w_1_dev,w_2_dev,w_3_dev

def update(a):#this is not right but FUCK IT DONT CARE
   return 1

######-----------------caculating the error and values--------------------######
