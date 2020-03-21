import perceptron as pc
import numpy as np

def main():
    print("loading training data...")
    load_data = np.array(np.genfromtxt("mnist_train.csv", delimiter=',', max_rows=1000))
    labels = load_data[:,0]
    training_data = np.delete(load_data, 0, 1)

    network = pc.network([784, 100, 10], ["sig", "sig"], training_data, labels)
    network.train(pc.warm_restarts(self, 0.0001, 0.1, 10, 2), 30)

main()
