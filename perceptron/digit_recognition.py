import perceptron as pc
import numpy as np

def mnist_load(file, samples):
    raw_data = np.array(np.genfromtxt(file, delimiter=',', max_rows=samples))
    labels = raw_data[:,0]
    data = np.delete(raw_data, 0, 1)*(1/255)
    return (data, labels)

def main():
    print("loading data...")
    samples = 10000
    batch_size = 20
    train = mnist_load("mnist_train.csv", samples)
    validate = mnist_load("mnist_test.csv", samples)

    structure = [784, 256, 128, 10, 10]
    activation_functions = ("relu", "relu", "relu", "softmax")
    restart_params = (0.0001, 0.01, .001, samples/(batch_size*2))

    network = pc.network(structure, activation_functions, train, validate)
    network.train(dropout=[.5,.5,0], beta=0.9, lr_func="warm restarts", lr_params=restart_params, batch_size=batch_size, epochs=20, cost_func="cross entropy")
main()
