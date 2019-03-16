import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Be careful with the file path!
data = loadmat('data/hw4.mat')
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(data['y'])

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# def ReLU(x):
#     return np.maximum(0.0001, x)

def forward_propagate(X, theta1, theta2):
    m = X.shape[0]

    #Write codes here

    # a1
    X_ndarray = X.A # turn the matrix to numpy ndarray
    bias = np.array(np.ones(m), ndmin=2) # create the 1*5000 bias list and turn the list to numpy ndarray(for concat)
    bias = bias.T # transpose
    X_ndarray = np.concatenate((bias, X_ndarray), axis=1)
    a1 = X_ndarray
    #print("a1 shape: ", a1.shape) -> (5000, 401)

    # z2
    z2 = np.matmul(a1, theta1.T)
    #print("z2 shape: ", z2.shape) -> (5000, 25)

    # a2
    a2 = sigmoid(z2)
    a2 = np.concatenate((bias, a2), axis=1)
    #print("a2 shape: ", a2.shape) -> 

    # z3
    z3 = np.matmul(a2, theta2.T)
    #print("z3 shape: ", z3.shape) -> (5000, 10)

    # h
    h = sigmoid(z3)
    #print("h shape: ", h.shape) -> 

    return a1, z2, a2, z3, h

def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    # run the feed-forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    # compute the cost
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        # np.place(h, h==0, [1.e-30])
        epsilon = 1e-30
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:] + epsilon))
        # second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)

    J = J / m
    J += (float(learning_rate) / (2*m) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2))))

    return J

# initial setup
input_size = 400 # input layer
hidden_size = 25 # hidden layer
num_labels = 10 # output layer
learning_rate = 0
# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.2
m = data['X'].shape[0] #
X = np.matrix(data['X'])
y = np.matrix(data['y'])
# unravel the parameter array into parameter matrices for each layer
theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):

    m = X.shape[0]

    #Write codes here

    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

    delta1 = np.zeros((hidden_size, input_size+1))
    delta2 = np.zeros((num_labels, hidden_size+1))

    theta1nobias = theta1[:, 1:]
    # print("theta1nobias shape: ", theta1nobias.shape) -> (25, 400)
    theta2nobias = theta2[:, 1:]
    # print("theta2nobias shape: ", theta2nobias.shape) -> (10, 25)

    for t in range(m):

        bias = np.array([1], ndmin=2)
        # print("bias shape: ", bias.shape) -> (1, 1)
        a1_ = np.concatenate((bias, X[t].T), axis=0) # (1*1) + (400*1) -> (401*1)
        # print("a1 shape: ", a1_.shape) -> (401, 1)
        z2_ = np.matmul(theta1, a1_) # (25*401) * (401*1) -> (25*1)
        # print("z2 shape: ", z2_.shape) -> (25, 1)
        a2_ = np.concatenate((bias, sigmoid(z2_)), axis=0) # (1*1) + (25*1) -> (26*1)
        # print("a2 shape: ", a2_.shape) -> (26, 1)
        z3_ = np.matmul(theta2, a2_) # (10*26) * (26*1) -> (10*1)
        # print("z3 shape: ", z3_.shape) -> (10, 1)
        a3_ = sigmoid(z3_) # (10*1) -> (10*1)

        # a1_, z2_, a2_, z3_, a3_ = forward_propagate(X[t], theta1, theta2)

        # print("a3 shape: ", a3_.shape) -> (10, 1)
        d3_ = np.subtract(a3_, y[t].reshape(10, 1)) # (10*1) (10*1) -> (10*1)
        # print("d3 shape: ", d3_.shape) -> (10, 1)
        d2_ = np.multiply(np.matmul(theta2nobias.T, d3_), sigmoid_gradient(z2_))   # (25*10) * (10*1) ? (25*1)
        # print("d2 shape: ", d2_.shape) -> (25, 1)
        delta2 = delta2 + np.matmul(d3_, a2_.T) # (10*26)
        # print("delta2 shape: ", delta2.shape) -> (10, 26)
        delta1 = delta1 + np.matmul(d2_, a1_.T) # (25*401)
        # print("delta1 shape: ", delta1.shape) -> (25, 401)

    theta1_grad = (1 / m) * delta1
    # print("theta1_grad shape: ", theta1_grad.shape) -> (25, 401)
    theta2_grad = (1 / m) * delta2
    # print("theta2_grad shape: ", theta2_grad.shape) -> (10, 26)

    theta1_grad[:, 1:] += ((learning_rate / m) * theta1nobias)
    theta2_grad[:, 1:] += ((learning_rate / m) * theta2nobias)
    # theta1_grad[:, 1:] += ((learning_rate / m) * theta1_grad[:, 1:])
    # theta2_grad[:, 1:] += ((learning_rate / m) * theta2_grad[:, 1:])

    grad = np.concatenate((theta1_grad.flatten(), theta2_grad.flatten()), axis=1)
    # grad = grad.A
    # grad = grad.flatten()
    J = cost(params, input_size, hidden_size, num_labels, X, y, learning_rate)

    return J, grad

from scipy.optimize import minimize, fmin_cg
# minimize the objective function
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate), method='TNC', jac=True, options={'maxiter': 1000, 'disp': True})
# fmin = fmin_cg(f=cost, x0=params, fprime=backprop, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate), maxiter=250, disp=True, full_output=False)
# print(fmin.shape)
# print(type(fmin))

print(fmin)

# method : TNC
    # Minimize a scalar function of one or more variables using a truncated Newton (TNC) algorithm.
    # truncated Newton algorithm :  optimizing non-linear functions with large numbers of independent variables.
    #                               The inner solver is truncated, i.e., run for only a limited number of iterations.

# jac : True
    # If jac is a Boolean and is True, fun is assumed to return the gradient along with the objective function.

# maxiter : int
    # Maximum number of iterations to perform.

X = np.matrix(X)
# theta1_ = np.matrix(np.reshape(fmin[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
# theta2_ = np.matrix(np.reshape(fmin[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
theta1_ = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2_ = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
a1, z2, a2, z3, h = forward_propagate(X, theta1_, theta2_)
y_pred = np.array(np.argmax(h, axis=1) + 1)



correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
accuracy = (sum(map(int, correct)) / float(len(correct)))
print ('accuracy = {0}%'.format(accuracy * 100))
