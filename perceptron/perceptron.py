"""
author: lincoln stefanello
"""
import numpy as np
import random
import activation_functions as af
import regularization as reg
import learning_rate as lr
import utils

class network:
    def __init__(self, parameters, act_funcs, training_data, labels):
       #a list describing the structure of the network. The length of the list is the number of layers, and each integer in the list is the number of neurons in the layer.
       self.parameters = parameters
       self.number_of_layers = len(self.parameters)
       self.act_funcs = act_funcs

       self.deck = list(zip(labels, training_data))
       random.shuffle(self.deck)
       self.labels, self.training_data = zip(*self.deck)
       self.desired_output = np.zeros(10)

       #sets the biases for each neuron in the network as zeros, skipping the first layer because it is the input layer.
       #Biases are indexed the same as neurons; e.g. the kth neuron in the ith layer has bias[i][k].
       self.biases = np.array([np.zeros((i, )) for i in self.parameters])

       #There is a weight matrix that exists between each layer. Therefore there are num_layers - 1 weight matrices.
       #This randomly initilizes each weight matrix and appends them to a list associated with layer;
       #e.g. W^i is the matrix of weights that connects the neurons in layer i with the neurons in layer i+1.
       #(w_j,k)^i is the weight connecting the jth neuron in layer i+1 to the kth neuron in layer i.
       #So, the activation a neuron k belonging to layer i takes in the vector of weights w[i-1][k]. (the kth row of the weight matrix between this layer and the previous).
       self.weights = np.array([np.random.randn(self.parameters[i+1], self.parameters[i])*np.sqrt(2/(self.parameters[i]+self.parameters[i+1])) for i in range(self.number_of_layers - 1)])

       #the z value is the dot product of weights & activations + bias quantity that gets sent to the squishification function.
       #The z value of the kth neuron in the ith layer is z_values[i][k].
       #the activation of the kth neuron in the ith layer is sigmoid(z_values[i][k]).
       self.z_values = np.array([np.empty((i, )) for i in self.parameters])
       self.z_primes = np.array([np.empty((i , )) for i in self.parameters])
       self.activations = np.array([np.empty((i, )) for i in self.parameters]) #the output of the network at each layer. The activation of the kth neuron in the ith layer is activations[i][k].
       self.learning_rate = 0
       self.index = 0 #which piece of training data the network is looking at

       self.size = training_data.shape[0] #the number of pieces of training data in an epoch
       self.trials = 1 #running total of pieces of training data the network has seen (resets every epoch)
       self.successes = 0 #running total of correct guesses (resets every epoch)
       self.cost = 0 #cost of an individual piece of training data
       self.costs = [] #used to compute average cost
       self.epoch = 1 #current epoch
       self.training = True

       #these are for minibatch gradient descent.
       #the batch_size is the size of random sample we take from the training data.
       #the batches hold the average gradient of each weight and bias over the random sample.
       #after the batch is exhausted, the true weights and biases are updated and the batches are reset.
       self.weights_batch = np.array([np.zeros((self.parameters[i+1], self.parameters[i])) for i in range(self.number_of_layers - 1)])
       self.biases_batch = np.array([np.zeros((i, )) for i in self.parameters])

       self.batch_clock = 0
       self.iterations = 0 #an "iteration" is the completion of a minibatch; e.g. if there are 10,000 pieces of training data and the minibatch is 20, then there are 500 iterations in an epoch.

    #looks at a piece of training data and assigns the activation kth neuron of the input layer to the kth pixel value of the piece of training data.
    def input_layer(self):
        self.activations[0] = list(zip(*self.deck))[1][self.index]

    #computes the activations of all the neurons in the layers subsequent to the input layer.
    #the kth row of matrix W^i contains every weight that attaches to the kth neuron in layer i+1,
    #therefore the z_value to be squished is just the dot product of the vector of activations from the previous layer
    #with the row vector (w_k)^(i-1) plus the bias.
    #to do this for an entire layer at a time, we matrix multiply W_(i-1) with activations and add the bias vector.
    def feedforward(self, dropout_rate=[]):
        for layer in range(1, self.number_of_layers): #iterates through every layer, skipping the input layer.
            activate = af.check_activation_function(self, layer, 0)
            #if dropout rates are not specified, default to having dropout turned off.
            #(Or, shut it off if we're in the output layer.)
            #the dropout list applies to the hidden layers, so the indexing is behind;
            #e.g. the 0th number in the dropout list is the dropout rate for the 1st layer.
            if (not dropout_rate) or (layer == self.number_of_layers - 1):
                self.z_values[layer] = np.dot(self.weights[layer-1], self.activations[layer-1]) + self.biases[layer]
                self.activations[layer] = activate(self.z_values[layer])
            else:
                self.activations[layer], self.z_values[layer] = reg.dropout(self, activate, layer, dropout_rate[layer-1])

    #computes the cost of an individual piece of training data and the overall average cost. Is this comment redundant?
    def compute_cost(self):
        object = int(list(zip(*self.deck))[0][self.index])
        self.desired_output = np.zeros(10)
        self.desired_output[object] = 1

        self.cost = (1/2)*np.linalg.norm(self.desired_output - self.activations[-1])**2 #individual cost for an individual piece of training data
        self.costs.append(self.cost) #used to compute the average cost.

    #tells us if the network is right or wrong.
    def evaluate(self):
        #the network's decision is the index of the highest number belonging to activation vector of the last layer.
        #if there are two elements that are both the highest, it chooses randomly between them.
        decision = np.random.choice(np.where(self.activations[-1] == np.amax(self.activations[-1]))[0])
        answer = int(list(zip(*self.deck))[0][self.index]) #extract the label of the training data to compare the decision with

        if (decision == answer):
            self.successes += 1 #woohoo!

        print("Epoch: ", self.epoch, " | ", self.successes, " correct out of ", self.trials, " for a rate of: ", \
         f"{(self.successes/self.trials)*100:.3f}%", " | ", "Guess: ", decision, " Answer: ", answer, " | ", "Avg cost: ", f"{sum(self.costs)/len(self.costs):.10f}", \
         " | ", "Learning rate: ", round(self.learning_rate, 5))

    #adjusts the learning rate depending on the learning rate function specified.
    def adjust_learning_rate(self, lr_func, lr_params, batch_size):
        if utils.batch_check(self, batch_size): #if the batch is depleted...
            #func_dict associates a string with a learning function via a dictionary.
            #we use the dictionary to look up the learning function we want, and send it the user's arguments.
            if (type(lr_params) != tuple):
                self.learning_rate = lr.func_dict()[lr_func](lr_params, batch_size)
            else:
                self.learning_rate = lr.func_dict()[lr_func](*lr_params, batch_size)

    #At the beginning of each epoch, the training data is shuffled, so simply portioning up its array into chunks of batch_size is taking a random sample.
    #If a batch size is not specified, it defaults to stochastic gradient descent (i.e. the batch_size is 1.)
    #batch_clock counts until we've exhausted the current batch.
    #When that happens, the weights and biases of the network are updated, and the weights and biases batches are reset.
    #This also implements momentum. If beta is not specified, momentum is turned off.
    def batch(self, batch_size, beta):
        if utils.batch_check(self, batch_size): #if the current batch is exhausted...
            self.iterations += 1
            if (beta != 0): #if momentum is enabled...
                print("uh, come back later!")
                #self.Vw = beta*self.Vw_prev + ((1 - beta)/batch_size)*self.weights_batch
                #self.Vb = beta*self.Vb_prev + ((1 - beta)/batch_size)*self.biases_batch

                #self.weights -= self.learning_rate*self.Vw
                #self.biases -= self.learning_rate*self.Vb

                #print(self.Vw)
            else:
                self.weights -= (self.learning_rate/batch_size)*self.weights_batch
                self.biases -= (self.learning_rate/batch_size)*self.biases_batch
            for i in range(self.number_of_layers - 1): self.weights_batch[i].fill(0); self.biases_batch[i].fill(0)
        self.batch_clock += 1

    #for every neuron in the network subsequent to the input layer, we adjust its weights and biases by a small amount proportional to the negative of d_C/d_w or dC_/d_b.
    def gradient_descent(self, neuron, layer):
        L = self.number_of_layers - 1
        dc_dal = sum(-1*(self.desired_output - self.activations[-1]))
        da_db = self.z_primes[layer][neuron]

        if (layer < L):
            dnext_dcurrent = np.prod([sum(np.dot(self.weights[layer+alpha].transpose(), self.z_primes[layer+alpha+1]) for alpha in range(L - layer))])

        else:
            dc_dal = -1*(self.desired_output[neuron] - self.activations[-1][neuron])
            dnext_dcurrent = 1

        gradient = da_db*dnext_dcurrent*dc_dal

        self.biases_batch[layer][neuron] += gradient
        self.weights_batch[layer-1][neuron] += np.array(self.activations[layer-1])*gradient

    #goes backwards through the network, sending each neuron it iterates through to the gradient descent function.
    def backprop(self):
        #the 0th layer of the reversed list is the last layer of the network. This makes my brain hurt, so  mirror flips the index sent to gradient_descent back to normal.
        mirror = self.number_of_layers - 1
        for layer in range(1, len(self.parameters)):
            activate = af.check_activation_function(self, layer, 1)
            self.z_primes[layer] = np.array(activate(self.z_values[layer]))

        backprop_layer = list(reversed(self.parameters[1:]))
        for layer in range(len(backprop_layer)):
            for neuron in list(reversed(range(backprop_layer[layer]))):
                self.gradient_descent(neuron, mirror)
            mirror -= 1

    #trains for how many epochs specified. rolls over the index of the data each new epoch.
    #reshuffles the data every epoch for GD.
    #resets the network's accuracy and avg cost
    #saves the weights and biases.
    def janitor(self, number_of_epochs):
        self.index += 1
        self.trials += 1

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
                 random.shuffle(self.deck)

    #beta is for smoothing
    def train(self, lr_func="constant", lr_params="(0.1)", batch_size=1, epochs=10, beta=0, dropout=[]):
        while self.training:
            self.adjust_learning_rate(lr_func, lr_params, batch_size)
            self.input_layer()
            self.feedforward(dropout)
            self.compute_cost()
            self.backprop()
            self.batch(batch_size, beta)
            self.evaluate()
            self.janitor(epochs)
