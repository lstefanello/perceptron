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

def check_activation_function(self, layer, order):
    if (order == 0):
        switcher = {
            "sig": np.vectorize(sigmoid),
            "relu": np.vectorize(relu),
            "l_relu": np.vectorize(leaky_relu),
            "tanh": np.vectorize(tanh),
        }
    elif (order == 1):
        switcher = {
            "sig": np.vectorize(sigmoid_prime),
            "relu": np.vectorize(relu_prime),
            "l_relu": np.vectorize(leaky_relu_prime),
            "tanh": np.vectorize(tanh_prime),
        }
    func = switcher.get(self.act_funcs[layer-1])
    return func
