from Lab1.NeuralNetwork import FeedForwardNet, DenseLayer
from Lab1 import Utils, Activations, Losses
import numpy as np


def generateNetwork(loss=Losses.MSE(), lr=0.001):
    layers = [DenseLayer(2, 1, Activations.Sigmoid())]
    return FeedForwardNet(layers, loss, learningRate=lr)


def getDecisionBoundry(xWeight, yWeight, bias):
    xMax = bias / xWeight
    yMax = bias / yWeight
    slope = -(yMax / xMax)
    return slope, bias


if __name__ == '__main__':
    p1, p2 = Utils.generateData(100, [[0, 0], [4, 4]], ['ro', 'bo'], [0.5, 0.5])
    Utils.plotPoints([p1, p2])
    model = generateNetwork()

    # Formatting to match input
    x1 = np.vstack([p1[0], p1[1]])
    x2 = np.vstack([p2[0], p2[1]])
    print("Traning data Shape:", x1.shape, x2.shape)
    # Generate Labels
    labels1 = np.ones((1, len(p1[0])))
    labels2 = np.zeros((1, len(p2[0])))
    print("Traning labels Shape:", labels1.shape, labels2.shape)

    x, y = Utils.stackAndShuffleData([x1, x2], [labels1, labels2])
    print("Final data shape:", x.shape, y.shape)
    for i in range(10000):
        print("Loss {}:".format(i), model.fit(x, y, batchSize=-1))

    w1, w2, b = model.layers[0].weights[0]
    print(getDecisionBoundry(w1, w2, b))
