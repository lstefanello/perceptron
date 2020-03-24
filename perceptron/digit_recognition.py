import perceptron as pc
import numpy as np

def main():
    print("loading training data...")
    #load training data from csv
    load_data = np.array(np.genfromtxt("mnist_train.csv", delimiter=',', max_rows=10000))
    #in the mnist data set, the first entry of every row is the label.
    #therefore, to get a vector of labels, we extract the first column.
    labels = load_data[:,0]
    #then we remove the labels from the data the network trains on
    training_data = np.delete(load_data, 0, 1)
    network = pc.network([784, 1000, 1000, 10], ("relu", "relu", "sigmoid"), training_data, labels)
    learn = (network, 0.0001, 0.001)
    #network.train(lr_func="warm restarts", lr_params=learn, batch_size=20, epochs=30)
    network.train(lr_func="const", lr_params=0.0001, dropout=[], batch_size=20, epochs=30)

main()
