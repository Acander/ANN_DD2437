from Lab1.NeuralNetwork import FeedForwardNet, DenseLayer
from Lab1 import Utils, Activations, Losses
import numpy as np


def generateNetwork(loss=Losses.MSE(), lr=0.001):
    layers = [
        DenseLayer(2, 15, Activations.ReLu()),
        DenseLayer(15, 1, Activations.Linear())
    ]
    return FeedForwardNet(layers, loss, learningRate=lr)


def _gaussFunc(x, y):
    return np.exp(-(x ** 2 + y ** 2) / 10) - 0.5


def generateTrainingData(numberOfPoints, min=-0.5, max=0.5):
    inData = np.random.random((numberOfPoints, 2)) * (max - min) + min
    labels = np.array([[_gaussFunc(x, y)] for x, y in inData])
    return inData.T, labels.T


model = generateNetwork()
print("My Model:", model)
inData, labels = generateTrainingData(100)
print("InData:", inData.shape, "  Labels:", labels.shape)

for i in range(1000):
    loss = model.fit(inData, labels)
    print(loss)
