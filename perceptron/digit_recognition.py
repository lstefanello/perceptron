import perceptron as pc
import cupy as cp

def mnist_load(file, samples):
    raw_data = cp.array(cp.genfromtxt(file, delimiter=',', max_rows=samples))
    labels = raw_data[:,0]
    data = cp.delete(raw_data, 0, 1)/255
    return (data, labels)

def main():
    print("loading data...")
    samples = 60000
    batch_size = 64
    train = mnist_load("mnist_train.csv", samples)
    validate = mnist_load("mnist_test.csv", None)

    structure = [784, 16, 16, 10]
    activation_functions = ("sigmoid", "sigmoid", "sigmoid")
    params = (0.1, 0.005)

    network = pc.network(structure, activation_functions, train, validate)
    network.train(dropout=[.5,0,0], beta=0.9, lr_func="exp decay", lr_params=params, batch_size=batch_size, epochs=20, cost_func="cross entropy")
main()
