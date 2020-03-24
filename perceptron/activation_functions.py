import numpy as np

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))

def tanh(z):
    return ((2 / (1 + np.exp(-2*z))) + 1)

def tanh_prime(z):
    return 1 - tanh(z)**2

def relu(z):
    if (z < 0):
        return 0
    elif (z >= 0):
        return z

def relu_prime(z):
    if (z < 0):
        return 0
    elif (z >= 0):
        return 1

def leaky_relu(z):
    if (z < 0):
        return 0.1*z
    elif (z >= 0):
        return z

def leaky_relu_prime(z):
    if (z < 0):
        return 0.1
    elif (z >= 0):
        return 1

def softmax(z, add):
    return np.exp(z)/add

def softmax_prime(z, add):
    return (np.exp(z)/add - (np.exp(z)/add)**2)

def func_dict(order):
    if (order == 0):
        dict = {
            "sigmoid": sigmoid,
            "relu": relu,
            "l_relu": leaky_relu,
            "tanh": tanh,
            "softmax": softmax,
        }
    elif (order == 1):
        dict = {
            "sigmoid": sigmoid_prime,
            "relu": relu_prime,
            "l_relu": leaky_relu_prime,
            "tanh": tanh_prime,
            "softmax": softmax_prime,
        }
    return dict
