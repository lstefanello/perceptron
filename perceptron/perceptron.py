"""
author: lincoln stefanello
"""
import numpy as np
import random
import activation_functions as af
import regularization as reg
import learning_rate as lr
import cost_functions as cf
import utils

class network:
    def __init__(self, parameters, act_funcs, training_data, testing_data):
       #a list describing the structure of the network. The length of the list is the number of layers, and each integer in the list is the number of neurons in the layer.
       self.parameters = parameters
       self.number_of_layers = len(self.parameters)
       self.act_funcs = act_funcs

       self.deck = list(zip(training_data[1], training_data[0]))
       random.shuffle(self.deck)
       self.labels, self.training_data = zip(*self.deck)
       self.desired_output = np.zeros((len(np.unique(self.labels)), ))

       self.validation_deck = list(zip(testing_data[1], testing_data[0]))
       self.validating = False

       #sets the biases for each neuron in the network as zeros, skipping the first layer because it is the input layer.
       #Biases are indexed the same as neurons; e.g. the kth neuron in the ith layer has bias[i][k].
       self.biases = np.array([np.full((i, ),0) for i in self.parameters])

       #There is a weight matrix that exists between each layer. Therefore there are num_layers - 1 weight matrices.
       #This randomly initilizes each weight matrix and appends them to a list associated with layer;
       #e.g. W^i is the matrix of weights that connects the neurons in layer i with the neurons in layer i+1.
       #(w_j,k)^i is the weight connecting the jth neuron in layer i+1 to the kth neuron in layer i.
       #So, the activation a neuron k belonging to layer i takes in the vector of weights w[i-1][k]. (the kth row of the weight matrix between this layer and the previous).
       self.weights = np.array([np.random.randn(self.parameters[i+1], self.parameters[i]) for i in range(self.number_of_layers - 1)])

       for matrix in range(self.number_of_layers - 1):
           switcher = {
               "sigmoid": np.sqrt(6/(self.parameters[matrix]+self.parameters[matrix+1])),
               "softmax": 0,
               "relu": np.sqrt(2/self.parameters[matrix]),
               "l_relu": np.sqrt(2/self.parameters[matrix]),
               "tanh": np.sqrt(6/(self.parameters[matrix]+self.parameters[matrix+1])),
               "elu": np.sqrt(1.55/(self.parameters[matrix]+self.parameters[matrix+1]))
               }
           self.weights[matrix] *= switcher.get(self.act_funcs[matrix])

       #the z value is the dot product of weights & activations + bias quantity that gets sent to the activation function.
       #The z value of the kth neuron in the ith layer is z_values[i][k].
       #the activation of the kth neuron in the ith layer is activation(z_values[i][k]).
       self.z_values = np.array([np.empty((i, )) for i in self.parameters])
       self.z_primes = np.array([np.empty((i , )) for i in self.parameters])
       self.activations = np.array([np.empty((i, )) for i in self.parameters]) #the output of the network at each layer. The activation of the kth neuron in the ith layer is activations[i][k].
       self.learning_rate = 0.1
       self.index = 0 #which piece of training data the network is looking at

       self.size = training_data[1].shape[0] #the number of pieces of training data in an epoch
       self.validation_size = testing_data[1].shape[0]
       self.trials = 1 #running total of pieces of training data the network has seen (resets every epoch)
       self.successes = 0 #running total of correct guesses (resets every epoch)
       self.costs = [] #used to compute average cost
       self.epoch = 1 #current epoch
       self.training = True

       #these are for minibatch gradient descent.
       #the batch_size is the size of random sample we take from the training data.
       #the batches hold the average gradient of each weight and bias over the random sample.
       #after the batch is exhausted, the true weights and biases are updated and the batches are reset.
       self.weights_batch = np.array([np.zeros((self.parameters[i+1], self.parameters[i])) for i in range(self.number_of_layers - 1)])
       self.biases_batch = np.array([np.zeros((i, )) for i in self.parameters])

       #these are for momentum.
       self.Vw = np.array([np.zeros((self.parameters[i+1], self.parameters[i])) for i in range(self.number_of_layers - 1)])
       self.Vb =  np.array([np.zeros((i, )) for i in self.parameters])

       self.batch_clock = 0
       self.iterations = 0 #an "iteration" is the completion of a minibatch; e.g. if there are 10,000 pieces of training data and the minibatch is 20, then there are 500 iterations in an epoch.
       self.total_iterations = 0 #doesn't reset every epoch

    #adjusts the learning rate depending on the learning rate function specified.
    def adjust_learning_rate(self, lr_func, lr_params, batch_size):
        if utils.batch_check(self, batch_size): #if the batch is depleted...
            #func_dict associates a string with a learning function via a dictionary.
            #we use the dictionary to look up the learning function we want, and send it the user's arguments.
            if (self.learning_rate <= 0.000001):
                self.learning_rate = 0.000001
            else:
                if (type(lr_params) != tuple):
                    self.learning_rate = lr.func_dict()[lr_func](self, lr_params)
                else:
                    self.learning_rate = lr.func_dict()[lr_func](self, *lr_params)

    #looks at a piece of training data and assigns the activation kth neuron of the input layer to the kth pixel value of the piece of training data.
    def input_layer(self):
        if not self.validating:
            self.activations[0] = np.array(list(zip(*self.deck))[1][self.index])
        else:
            self.activations[0] = np.array(list(zip(*self.validation_deck))[1][self.index])

    #computes the activations of all the neurons in the layers subsequent to the input layer.
    #the kth row of matrix W^i contains every weight that attaches to the kth neuron in layer i+1,
    #therefore the z_value to be activated is just the dot product of the vector of activations from the previous layer
    #with the row vector (w_k)^(i-1) plus the bias.
    #to do this for an entire layer at a time, we matrix multiply W_(i-1) with activations and add the bias vector.
    def feedforward(self, dropout_rate=[]):
        for layer in range(1, self.number_of_layers): #iterates through every layer, skipping the input layer.
            if (layer == self.number_of_layers - 1 and self.act_funcs[-1] == "softmax"): #softmax has no weights or biases.
                self.z_values[-1] = self.activations[-2] - self.activations[-2].max() #computational trick to avoid overflow.
                self.activations[-1] = af.softmax(self.z_values[-1])
            else:
                activate = af.func_dict(0)[self.act_funcs[layer-1]]
                if (self.validating == True):
                    if (not dropout_rate) or (dropout_rate[layer-1] == 0):
                        scale = 1
                    else:
                        scale = (1 - dropout_rate[layer-1]) #present with probability 1-p.
                    self.z_values[layer] = np.dot(self.weights[layer-1]*scale, self.activations[layer-1]) + self.biases[layer]
                    self.activations[layer] = activate(self.z_values[layer])
                else:
                    self.z_values[layer] = np.matmul(self.weights[layer-1], self.activations[layer-1]) + self.biases[layer]
                    self.activations[layer] = activate(self.z_values[layer])
                    if (dropout_rate) and (dropout_rate[layer-1] != 0):
                        self.activations[layer], self.z_values[layer] = reg.dropout(self.activations[layer], self.z_values[layer], dropout_rate[layer-1], self.parameters[layer])

    #computes the cost of an individual piece of training data and the overall average cost. Is this comment redundant?
    def compute_cost(self, cost_func):
        if not self.validating:
            object = int(list(zip(*self.deck))[0][self.index])
        else:
            object = int(list(zip(*self.validation_deck))[0][self.index])

        self.desired_output.fill(0)
        self.desired_output[object] = 1
        self.costs.append(cf.func_dict(0)[cost_func](self)) #used to compute the average cost.

    #tells us if the network is right or wrong.
    def evaluate(self):
        #the network's decision is the index of the highest number belonging to activation vector of the last layer.
        #if there are two elements that are both the highest, it chooses randomly between them (probably doesn't ever happen).
        decision = np.random.choice(np.where(self.activations[-1] == np.amax(self.activations[-1]))[0])

        if not self.validating:
            answer = int(list(zip(*self.deck))[0][self.index]) #extract the label of the training data to compare the decision with
        else:
            answer = int(list(zip(*self.validation_deck))[0][self.index])

        if (decision == answer):
            self.successes += 1 #woohoo!

        if self.validating:
            print("Validation", " | ", self.successes, " correct out of ", self.trials, " for a rate of: ", \
             f"{(self.successes/self.trials)*100:.3f}%", " | ", "Guess: ", decision, " Answer: ", answer, " | ", "Avg cost: ", f"{sum(self.costs)/len(self.costs):.10f}")
        else:
            print("Epoch: ", self.epoch, " | ", self.successes, " correct out of ", self.trials, " for a rate of: ", \
             f"{(self.successes/self.trials)*100:.3f}%", " | ", "Guess: ", decision, " Answer: ", answer, " | ", "Avg cost: ", f"{sum(self.costs)/len(self.costs):.10f}", \
             " | ", "Learning rate: ", round(self.learning_rate, 5))

    #At the beginning of each epoch, the training data is shuffled, so simply portioning up its array into chunks of batch_size is taking a random sample.
    #If a batch size is not specified, it defaults to stochastic gradient descent (i.e. the batch_size is 1.)
    #batch_clock counts until we've exhausted the current batch.
    #When that happens, the weights and biases of the network are updated, and the weights and biases batches are reset.
    #This also implements momentum. If beta is not specified, momentum is turned off.
    def batch(self, batch_size, beta):
        if utils.batch_check(self, batch_size): #if the current batch is exhausted...
            self.iterations += 1
            self.total_iterations += 1
            if (beta != 0): #if momentum is enabled...
                self.Vw = beta*self.Vw + self.weights_batch/batch_size
                self.Vb = beta*self.Vb + self.biases_batch/batch_size

                self.weights -= self.learning_rate*self.Vw
                self.biases -= self.learning_rate*self.Vb
            else:
                self.weights -= (self.learning_rate/batch_size)*self.weights_batch
                self.biases -= (self.learning_rate/batch_size)*self.biases_batch

            for i in range(len(self.weights_batch)): self.weights_batch[i].fill(0)
            for i in range(len(self.biases_batch)): self.biases_batch[i].fill(0)
        self.batch_clock += 1


    def backprop(self, cost_func):
        L = self.number_of_layers - 1
        gradient = cf.func_dict(1)[cost_func](self) #the change in the user-specified cost with respect to the activation of the last layer
        self.z_primes[L] = af.func_dict(1)[self.act_funcs[-1]](self.z_values[L])
        self.z_primes[L-1] = af.func_dict(1)[self.act_funcs[-2]](self.z_values[L-1])

        if (self.act_funcs[-1] != "softmax"): #the softmax layer doesn't have weights or biases, so we don't bother with trying to update them.
            self.biases_batch[L] += gradient*self.z_primes[L]
            self.weights_batch[L-1] += np.outer(gradient*self.z_primes[L], self.activations[L-1])

        for distance in range(L-1):
            if (self.act_funcs[-1] == "softmax" and distance == 0):
                gradient = af.softmax_input_change(self.z_values[L])*self.z_primes[L]*gradient
            else:
                self.z_primes[L-distance] = af.func_dict(1)[self.act_funcs[-1-distance]](self.z_values[L-distance])
                self.z_primes[L-distance-1] = af.func_dict(1)[self.act_funcs[-2-distance]](self.z_values[L-distance-1])
                gradient = np.dot(self.weights[L-distance-1].transpose(), self.z_primes[L-distance]*gradient)

            self.biases_batch[L-distance-1] += self.z_primes[L-distance-1]*gradient
            self.weights_batch[L-distance-2] += np.outer(self.z_primes[L-distance-1]*gradient, self.activations[L-distance-2])


    #    for i in range(len(self.weights_batch)-1): print(self.weights_batch[i], i)
    #trains for how many epochs specified. rolls over the index of the data each new epoch.
    #reshuffles the data every epoch for GD.
    #resets the network's accuracy and avg cost
    #saves the weights and biases.
    def janitor(self, number_of_epochs):
        self.index += 1
        self.trials += 1

        if not self.validating:
            if (self.index > self.size - 1):
                self.epoch += 1
                self.index = 0
                self.trials = 1
                self.successes = 0
                self.iterations = 0
                self.costs.clear()

                for i in range(len(self.weights)):
                    np.savetxt("weights{}.csv".format(i), self.weights[i], delimiter=",")

                for i in range(1, len(self.biases)):
                    np.savetxt("biases{}.csv".format(i), self.biases[i], delimiter=",")

                if(self.epoch > number_of_epochs):
                    self.training = False
                else:
                    self.validating = True
                    random.shuffle(self.deck)
        else:
            if (self.index > self.validation_size - 1):
                self.index = 0
                self.trials = 1
                self.successes = 0
                self.costs.clear()
                self.validating = False

    def train(self, lr_func="constant", lr_params=0.0001, batch_size=1, epochs=10, beta=0, dropout=[], cost_func="quadratic"):
        while self.training:
            if self.validating:
                self.input_layer()
                self.feedforward(dropout)
                self.compute_cost(cost_func)
                self.evaluate()
                self.janitor(epochs)
            else:
                self.adjust_learning_rate(lr_func, lr_params, batch_size)
                self.input_layer()
                self.feedforward(dropout)
                self.compute_cost(cost_func)
                self.backprop(cost_func)
                self.batch(batch_size, beta)
                self.evaluate()
                self.janitor(epochs)
