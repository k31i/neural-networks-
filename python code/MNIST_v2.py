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

X_train = np.split(X_train, X_train.shape[1], axis=1)

#-------define the pramaters aka weights and nodes


weig_1 = np.random.uniform(low=-2, high=2, size=(16, 784))
weig_2 = np.random.uniform(low=-2, high=2, size=(16, 16))
weig_3 = np.random.uniform(low=-2, high=2, size=(10, 16))

bias_1 = np.random.uniform(low=-1,high=1,size=(16,1))
bias_2 = np.random.uniform(low=-1,high=1,size=(16,1))
bias_3 = np.random.uniform(low=-1,high=1,size=(10,1))

#-------define the functions to be used. so forwarad and back propagation--------------#



def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def softactive(x):
   return np.exp(x) / sum(np.exp(x))

def one_hot(Y):#################################coppied from kaggel and used in back prop
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def forward_prop(a_0,w_1,w_2,w_3,b_1,b_2,b_3):#correct for dot prod
   a_0 = np.hstack([np.reshape(mat, (784, 1)) for mat in a_0])#chat GPT told me to
   z_1 = np.dot(w_1,a_0) - b_1
   a_1 = sigmoid(z_1)
   print(a_0.shape)
   print(w_1.shape)
   print(z_1.shape)
   print(a_1.shape)
   
   z_2 = w_2.dot(a_1) - b_2
   a_2 = sigmoid(z_2)

   z_3 = w_3.dot(a_2) - b_3 
   a_3 = softactive(z_3)
   return z_1,z_2,z_3,a_1,a_2,a_3

def sigmoid_dev(x):
   return (1/(1+np.e**(-x)))*(1-(1/(1+np.e**(-x))))

def softactive_dev(x):#FUCK THIS!!!
   return 1

def back_prop(Y,a_3,a_2,a_1,a_0,z_3,z_2,z_1,w_3,w_2):#dont know if .T is right i just put them in to avoid eorr but need to figure out why
   si_3 = np.dot(2*(a_3-Y),softactive_dev(z_3))
   b_3_dev = si_3
   w_3_dev = np.dot(si_3,a_2.T)
   si_2 = np.dot(np.dot(np.transpose(w_3),si_3),sigmoid_dev(z_2).T)
   b_2_dev = si_2
   print(si_2.shape)
   print(a_1.shape)
   w_2_dev = np.dot(si_2,a_1)
   si_1 = np.dot(np.dot(np.transpose(w_2),si_2),sigmoid_dev(z_1))
   b_1_dev = si_1
   a_0 = np.hstack([np.reshape(mat, (784, 1)) for mat in a_0])#chat GPT told me to
   w_1_dev = np.dot(si_1,a_0.T)
   return b_1_dev,b_2_dev,b_3_dev,w_1_dev,w_2_dev,w_3_dev

def update(w1,w2,w3,dw1,dw2,dw3,b1,b2,b3,db1,db2,db3,alpha):
    w1 = w1 - alpha * dw1
    b1 = b1 - alpha * db1
    w2 = w2 - alpha * dw2
    b2 = b2 - alpha * db2
    w3 = w3 - alpha * dw3
    b3 = b3 - alpha * db3
    return w1,w2,w3,b1,b2,b3

######-----------------caculating the error and values--------------------######

#straight up cntrl+c / cntrl+v this stuff

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    global weig_1, bias_1, weig_2, bias_2, weig_3, bias_3
    for i in range(iterations):
        Z1,Z2,Z3,A1,A2,A3 = forward_prop(X,weig_1,weig_2,weig_3,bias_1,bias_2,bias_3)
        db1, db2, db3, dW1, dW2, dW3 = back_prop(Y,A3,A2,A1,X,Z3,Z2,Z1,weig_3,weig_2)
        weig_1, weig_2, weig_3, bias_1, bias_2, bias_3 = update(weig_1, weig_2, weig_3, dW1, dW2, dW3, bias_1, bias_2, bias_3, db1, db2, db3, alpha)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return weig_1,bias_1,weig_2,bias_2,weig_3,bias_3

W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 0.10, 500)