import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def inital_paramaters():
    weig_1 = np.random.uniform(low=-2, high=2, size=(16, 784))
    weig_2 = np.random.uniform(low=-2, high=2, size=(16, 16))
    weig_3 = np.random.uniform(low=-2, high=2, size=(10, 16))
    
    bias_1 = np.random.uniform(low=-1,high=1,size=(16,1))
    bias_2 = np.random.uniform(low=-1,high=1,size=(16,1))
    bias_3 = np.random.uniform(low=-1,high=1,size=(10,1))
    return weig_1,weig_2,weig_3,bias_1,bias_2,bias_3

def sigmoid(x):
  # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def sigmoid_derivative(z):
   return sigmoid(z)*(1-sigmoid(z))

def softactive(x):
   #for last layer
   return np.exp(x) / sum(np.exp(x))

def forward_prop(X,W1,W2,W3,B1,B2,B3):
   Z1 = np.dot(W1,X)+B1
   A1 = sigmoid(Z1)
   Z2 = np.dot(W2,A1)+B2
   A2 = sigmoid(Z2)
   Z3 = np.dot(W3,A2)+B3
   A3 = softactive(Z3)
   return A3,A2,A1,Z3,Z2,Z1

def back_prop(X,Y,W1,W2,W3,A1,A2,A3,Z1,Z2,Z3):
   cost = (A3-Y)
   delta_3 = 2*cost*sigmoid_derivative(Z3)
   delta_b_3 = delta_3
   delta_w_3 = np.dot(delta_3,A2.transpose())
   delta_2 = (np.dot(W3.transpose(),delta_3))*sigmoid_derivative(Z2)
   delta_b_2 = delta_2
   delta_w_2 = np.dot(delta_2,A1.transpose())
   delta_1 = (np.dot(W3.transpose(),delta_2))*sigmoid_derivative(Z1)
   delta_b_1 = delta_1
   delta_w_1 = np.dot(delta_1,X.transpose())
   return delta_w_1,delta_w_2,delta_w_3,delta_b_1,delta_b_2,delta_b_3

def correction(alpha,w1,dw1,w2,dw2,w3,dw3,b1,db1,b2,db2,b3,db3):
   w1 = w1 - (alpha*dw1)
   w2 = w2 - (alpha*dw2)
   w3 = w3 - (alpha*dw3)
   b1 = b1 - (alpha*db1)
   b2 = b2 - (alpha*db2)
   b3 = b3 - (alpha*db3)
   return w1,w2,w3,b1,b2,b3
   