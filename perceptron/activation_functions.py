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
    def comp(x):
        if (x >= 0):
            return x
        else:
            return 0
    return np.array([comp(i) for i in z])

def relu_prime(z):
    def comp(x):
        if (x < 0):
             return 0
        elif (x >= 0):
            return 1

    return np.array([comp(i) for i in z])

def leaky_relu(z):
    def comp(x):
        if (x < 0):
            return 0.1*x
        elif (x >= 0):
            return x
    return np.array(comp(i) for i in z)

def leaky_relu_prime(z):
    def comp(x):
        if (x < 0):
            return 0.1
        elif (x >= 0):
            return 1
    return np.array(comp(i) for i in z)

def softmax(z):
    return np.exp(z)/sum(np.exp(z))

def softmax_prime(z):
    #this is the diagonal of the jacobian.
    return softmax(z)*(1-softmax(z))

def softmax_input_change(z):
    dz_da = np.zeros((z.shape[0], z.shape[0]))
    for i in range(len(z)):
        tmp = []
        if (z[i] == 0):
            for j in z:
                if (j == 0):
                    tmp.append(0)
                else:
                    tmp.append(-1)
        else:
            for j in range(len(z)):
                if (i == j):
                    tmp.append(1)
                else:
                    tmp.append(0)

        dz_da[:,i] += np.array(tmp)

    return dz_da.diagonal()

def func_dict(order):
    if (order == 0):
        dict = {
            "sigmoid": sigmoid,
            "relu": relu,
            "l_relu": leaky_relu,
            "tanh": tanh,
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
