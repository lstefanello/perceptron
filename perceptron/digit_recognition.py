import perceptron as pc
import numpy as np

def main():
    print("loading training data...")
    load_data = np.array(np.genfromtxt("mnist_train.csv", delimiter=',', max_rows=5000))
    labels = load_data[:,0]
    training_data = np.delete(load_data, 0, 1)

    network = pc.network([784, 800, 10], ("sig", "sig"), training_data, labels)
    learn = (network, 0.001, 0.1)
    network.train(lr_func="warm restarts", lr_params=learn, batch_size=20, epochs=30)

main()
