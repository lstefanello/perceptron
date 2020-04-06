import numpy as np

def quadratic(model):
    return sum((model.activations[-1] - model.desired_output)**2)

def quadratic_derivative(model):
    return 2*(model.activations[-1] - model.desired_output)

def cross_entropy(model):
    log = np.vectorize(np.log)
    return np.dot(-1*model.desired_output, log(model.activations[-1]))

def cross_entropy_derivative(model):
    return -1*model.desired_output/model.activations[-1]

def func_dict(order):
    if (order == 0):
        dict = {
            "quadratic": quadratic,
            "cross entropy": cross_entropy
            }
    elif (order == 1):
        dict = {
            "quadratic": quadratic_derivative,
            "cross entropy": cross_entropy_derivative
        }
    return dict
