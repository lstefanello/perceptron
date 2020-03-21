import numpy as np

#randomly shuts of neurons from each hidden layer (i.e., forces them to output zero) to combat overfitting.
def dropout(self, activate, layer, dropout_rate):
    turn_off = int(np.floor(dropout_rate*self.parameters[layer])) #determines how many neurons from the layer to shut off
    picks = [np.random.randint(0, self.parameters[layer]) for i in range(turn_off)] #randomly picks that number of neurons
    z = np.dot(self.weights[layer-1], self.activations[layer-1]) + self.biases[layer] #load in the z-vals of the current layer
    a = activate(z) #send to squishification
    for i in picks: a[i] = 0 #zeros out the activations we chose
    return a, z #returns zero'd-out activations, then normal z-vals.
