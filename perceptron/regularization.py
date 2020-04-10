import cupy as cp
import random

#randomly shuts of neurons from each hidden layer (i.e., forces them to output zero) to combat overfitting.
def dropout(a, z, dropout_rate, neurons):
    turn_off = int(cp.floor(dropout_rate*neurons)) #determines how many neurons from the layer to shut off
    picks = random.sample(range(neurons), turn_off) #randomly picks that number of neurons
    for i in picks: a[i] = 0; z[i] = 0 #zeros out the activations we chose
    return a, z
