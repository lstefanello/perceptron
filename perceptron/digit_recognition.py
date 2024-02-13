import perceptron as pc
import numpy as np

def mnist_load(file, samples):
    raw_data = np.array(np.genfromtxt(file, delimiter=',', max_rows=samples))
    labels = raw_data[:,0]
    data = np.delete(raw_data, 0, 1)/255.0
    return (data, labels)

def main():
    print("loading data...")
    samples = 10000
    batch_size = 32
    train = mnist_load("mnist_train.csv", samples)
    validate = mnist_load("mnist_test.csv", samples)

    structure = [784, 512, 512, 10, 10]
    activation_functions = ("relu", "relu", "relu", "softmax")
    network = pc.network(structure, activation_functions, train, validate)
    network.train(dropout=[0, 0, 0], beta=0.75, lr_func="const", lr_params=0.1, batch_size=batch_size, epochs=5, cost_func="cross entropy")

main()
