import numpy

#At the beginning of each epoch, the training data is shuffled, so simply portioning up its array into chunks of batch_size is taking a random sample.
#If a batch size is not specified, it defaults to stochastic gradient descent (i.e. the batch_size is 1.)
#batch_clock counts until we've exhausted the current batch.
#When that happens, the weights and biases of the network are updated, and the weights and biases batches are reset.
#This also implements momentum. If beta is not specified, momentum is turned off.
def gd_batch(self, batch_size, beta):
    if utils.batch_check(self, batch_size): #if the current batch is exhausted...
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

#order = 0 for the optimizer itself, order = 1 for its batching.
def func_dict(order):
    if (order == 0):
        dict = {
            "gd": gradient_descent,
        }
    elif (order == 1):
        dict = {
            "gd": gd_batch
        }
