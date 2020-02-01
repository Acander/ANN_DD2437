from Lab1.NeuralNetwork import FeedForwardNet, DenseLayer
from Lab1 import Utils, Activations, Losses
import numpy as np


def generateNetwork(hidden, loss=Losses.MSE(), lr=0.0015):
    layers = [
        DenseLayer(2, hidden, Activations.Sigmoid()),
        DenseLayer(hidden, hidden, Activations.ReLu()),
        DenseLayer(hidden, 1, Activations.Linear())
    ]
    return FeedForwardNet(layers, loss, learningRate=lr)


def _gaussFunc(x, y):
    return np.exp(-(x ** 2 + y ** 2) / 10) - 0.5


def generateTrainingData(numberOfPoints, min=-0.5, max=0.5):
    inData = np.random.random((numberOfPoints, 2)) * (max - min) + min
    labels = np.array([[_gaussFunc(x, y)] for x, y in inData])
    return inData.T, labels.T


def splitIntoEval(x, y, trainingRatio=0.6):
    t = int(x.shape[1] * trainingRatio)
    return x[:, :t], y[:, :t], x[:, t:], y[:, t:]


inData, labels = generateTrainingData(5000)

batchsizes = [-1, 128, 64, 1]
inData, labels = Utils.shuffleData(inData, labels)
inTrain, labelTrain, inTest, labelTest = splitIntoEval(inData, labels)
allModelLosses = []
model = generateNetwork(64)
print("My Model:", model)

for i in range(200):
    inTrain, labelTrain = Utils.shuffleData(inTrain, labelTrain)
    print(i, model.fit(inTrain, labelTrain, batchSize=16))

output = model.forwardPass(inTest)
print(inTest.shape, output.shape)
points = [(inTest[0], inTest[1], output[0], 'ro')]

Utils.plot3DMeshgridGaussianSamples(points)
'''
for b in batchsizes:
    model = generateNetwork(8)
    losses = []
    for i in range(20):
        fullLoss = model.loss.forward(model.forwardPass(inData), labels)
        losses.append([fullLoss])

        model.fit(inTrain, labelTrain, batchSize=b)

    allModelLosses.append(losses)

top = max([max([i for i in l]) for l in losses])
for l in allModelLosses:
    print(l)
    l.insert(0, [top])

colors = ['red', 'blue', 'green', 'orange']
plots = []
for l, c, b in zip(allModelLosses, colors, batchsizes):
    plots.append((np.arange(len(l)), l, c, b))

print(len(plots), len(plots[0]))

Utils.plotPoints(plots)
'''
