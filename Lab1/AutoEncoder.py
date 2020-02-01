from Lab1.NeuralNetwork import FeedForwardNet, DenseLayer
from Lab1 import Utils, Activations, Losses
import numpy as np
import pickle


def generateNetwork(loss=Losses.MSE(), lr=0.001):
    layers = [
        DenseLayer(8, 2, Activations.Linear()),
        DenseLayer(2, 8, Activations.Sigmoid())
    ]
    return FeedForwardNet(layers, loss, learningRate=lr)


def generateTrainingData(numberOfPoints, size):
    inData = np.zeros((size, numberOfPoints))
    for i in range(numberOfPoints):
        inData[np.random.randint(size)][i] = 1
    return inData


if __name__ == '__main__':
    inData = generateTrainingData(400, 8)
    print("InData:", inData.shape)
    model = generateNetwork()
    for i in range(50000):
        loss = model.fit(inData, inData, batchSize=64)
        if (i % 100 == 0):
            print(i, loss)

    weights = [l.weights for l in model.layers]
    with open('ModelWeights2.pkl', 'wb') as fp:
        pickle.dump(weights, fp)
    '''
    '''
