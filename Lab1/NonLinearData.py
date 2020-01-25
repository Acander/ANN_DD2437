from Lab1.NeuralNetwork import FeedForwardNet, DenseLayer
from Lab1 import Utils, Activations, Losses
import numpy as np


def generateNetwork(loss=Losses.MSE(), lr=0.001):
    layers = [
        DenseLayer(2, 2, Activations.ReLu()),
        DenseLayer(2, 1, Activations.Sigmoid())
    ]
    return FeedForwardNet(layers, loss, learningRate=lr)


p1, p2, p3 = Utils.generateData(100, [[-4, -4], [0, 0], [4, 4]], ['bo', 'ro', 'bo'], [0.5, 0.5, 0.5])
Utils.plotPoints([p1, p2, p3])
model = generateNetwork()
print("My model:", model)

inData = [np.vstack([p[0], p[1]]) for p in [p1, p2, p3]]
labels = [np.ones((1, len(p1[0]))), np.zeros((1, len(p1[0]))), np.ones((1, len(p1[0])))]

x, y = Utils.stackAndShuffleData(inData, labels)
print("Indata:", x.shape, " Labels:", y.shape)
for i in range(10000):
    print("Loss {}:".format(i), model.fit(x, y, batchSize=10))
