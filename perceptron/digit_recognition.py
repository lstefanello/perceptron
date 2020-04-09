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
    train = mnist_load("mnist_train.csv", samples)
    validate = mnist_load("mnist_test.csv", np.floor(samples/6))

    structure = [784, 512, 10, 10]
    activation_functions = ("relu", "relu", "softmax")
    step_params = (samples/40, 0.01, 1/2)
    #restart_params = (0.01, 0.001, 3/4)

    network = pc.network(structure, activation_functions, train, validate)
    #network.train(dropout=[.5,0], beta=0.9, lr_func="warm restarts", lr_params=restart_params, batch_size=20, epochs=30, cost_func="cross entropy")
    #network.train(dropout=[.5,.3,0], beta=0.9, lr_func="step decay", lr_params=step_params, batch_size=20, epochs=30, cost_func="cross entropy")
    network.train(dropout=[.5,0], beta=0.9, lr_func="const", lr_params=0.0001, batch_size=20, epochs=30, cost_func="cross entropy")

main()
