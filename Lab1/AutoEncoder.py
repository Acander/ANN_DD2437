from Lab1.NeuralNetwork import FeedForwardNet, DenseLayer
from Lab1 import Utils, Activations, Losses
import numpy as np
import pickle


def generateNetwork(loss=Losses.MSE(), lr=0.001):
    layers = [
        DenseLayer(8, 3, Activations.ReLu()),
        DenseLayer(3, 8, Activations.Sigmoid())
    ]
    return FeedForwardNet(layers, loss, learningRate=lr)


def generateTrainingData(numberOfPoints, size):
    inData = np.zeros((size, numberOfPoints))
    for i in range(numberOfPoints):
        inData[np.random.randint(size)][i] = 1
    return inData


inData = generateTrainingData(400, 8)
print("InData:", inData.shape)
model = generateNetwork()
for i in range(60000):
    loss = model.fit(inData, inData, batchSize=64)
    if (i % 100 == 0):
        print(i, loss)

'''
weights = [l.weights for l in model.layers]
with open('ModelRandomWeights.pkl', 'wb') as fp:
    pickle.dump(weights, fp)
'''

print(model.layers[0].weights.T.shape)
points = []
for p in model.layers[0].weights.T:
    points.append([[c] for c in p] + ['ro'])
print(points)
Utils.plot3D(points)
