#!/usr/bin/env python
# coding: utf-8

# In[306]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat


# In[307]:


data = loadmat('data/hw4.mat')
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(data['y'])


# In[308]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# In[309]:


def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    #Write codes here
    
    X_ndarr = np.matrix(X)
    tmp=np.ones(m)
    bias = np.array(tmp, ndmin=2)
    X_ndarr = np.concatenate((bias.T, X_ndarr), axis=1)
    a1= X_ndarr
    
    z2= np.matmul(a1, theta1.T)
    
    a2= sigmoid(z2)
    a2= np.concatenate((bias.T, a2), axis=1)
    
    z3= np.matmul(a2, theta2.T)
    
    h= sigmoid(z3)
    
    return a1, z2, a2, z3, h


# In[310]:


def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    #print("I am in cost function 1")
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    #print("I am in cost function 2")
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    # run the feed-forward pass
    #print("I am in cost function 3")
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # compute the cost
    J = 0
    #print("I am in cost function 4")
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        epsilon = 1e-30
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
    #print("I am in cost function 5")
    J = J / m
   	#J += (float(learning_rate) / (2*m) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:]))))
    J += (float(learning_rate) / (2*m) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2))))

    #print("I am in cost function 6")

    return J


# In[311]:


input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1
# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.2
m = data['X'].shape[0]
X = np.matrix(data['X'])
y = np.matrix(data['y'])
# unravel the parameter array into parameter matrices for each layer
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)


# In[312]:


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))    


# In[313]:


def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    
    #Write codes here
    
    theta1_b= np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2_b= np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    
    delta1= np.zeros((hidden_size, input_size+1))
    delta2= np.zeros((num_labels, hidden_size+1))
    
    theta1= theta1_b[:, 1:]
    #print("theta1withoutbias shape: ", theta1_b.shape) #-> (25, 400)
    theta2= theta2_b[:, 1:]
    #print("theta2withoutbias shape: ", theta2_b.shape) #-> (10, 25)
    
    for i in range(m):
        
        bias=np.array([1], ndmin=2)
        
        a1_1=np.concatenate((bias, X[i].T), axis=0)
        z2_2=np.matmul(theta1_b, a1_1)
        a2_2=np.concatenate((bias, sigmoid(z2_2)), axis=0)
        z3_3=np.matmul(theta2_b, a2_2)
        a3_3=sigmoid(z3_3)
        
        d3= np.subtract(a3_3, y[i].reshape(10, 1))
        
        d2= np.multiply(np.matmul(theta2.T, d3), sigmoid_gradient(z2_2))
            
        delta2 = delta2 + np.matmul(d3, a2_2.T)
        delta1 = delta1 + np.matmul(d2, a1_1.T)
        
    theta1_grad = (1 / m) * delta1
    theta2_grad = (1 / m) * delta2
        
    theta1_grad[:, 1:] += ((learning_rate / m) * theta1)
    theta2_grad[:, 1:] += ((learning_rate / m) * theta2)
        
        
    J=cost(params, input_size, hidden_size, num_labels, X, y, learning_rate)
    grad=np.concatenate((theta1_grad.flatten(), theta2_grad.flatten()), axis=1)
        
    return J, grad


# In[ ]:


from scipy.optimize import minimize
# minimize the objective function
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate), method='TNC', jac=True, options={'maxiter':800})
 
X = np.matrix(X)
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
y_pred = np.array(np.argmax(h, axis=1) + 1)

correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))


# In[ ]:





# In[ ]:





# In[ ]:




