import numpy as np

from Lab1.LinearSeparableData import getDecisionBoundry
from Lab1.Utils import plotPoints, generateData, stackAndShuffleData


class PLNet:

    def __init__(self, learningRate, sizeIn):
        self.lr = learningRate
        self.weights = np.random.normal(0, 0.5, sizeIn + 1)

    def predictRaw(self, x):
        return np.matmul(self.weights, x)

    def predict(self, x):
        # self.weights @ np.vstack([x, ([1] * np.size(x[0]))])
        return ((np.matmul(self.weights, x) > 0).astype(int) * 2) - 1  # convert to -1/1

    # Assumes labels are -1 or 1
    def fit(self, x, labels):
        '''
        predictions = self.predict(x)
        falsePredictions = np.invert(np.equal(predictions, labels).astype(int))
        # Only weights with incorrect prediction will be changed
        # Increases weight if prediction was negative and vice versa
        self.weights += self.lr * falsePredictions * (predictions * -1) * np.transpose(x)
        '''
        for i in range(np.size(labels)):
            xCol = x[:, i]  # one data point is one column
            prediction = self.predict(xCol)
            falsePrediction = np.invert(np.equal(prediction, labels[i])).astype(int)
            self.weights += self.lr * falsePrediction * (prediction * -1) * np.transpose(xCol)


def getTrainedModel(x, y):
    perceptron = PLNet(0.001, 2)
    print(perceptron.weights)

    N_EPOCHS = 50
    for i in range(N_EPOCHS):
        if i % 5 == 0:
            print("Accuracy:", getAccuracy(perceptron, x, y))
        perceptron.fit(x, y)

    return perceptron


def getAccuracy(model, x, y):
    predictions = model.predict(x)
    numCorrect = np.sum(np.equal(predictions, y).astype(int))
    freqCorr = numCorrect / (N_CLASS * 2)
    return freqCorr


def generateStructureShuffleData(n_class, centroids, stdDevs):
    p1, p2 = generateData(n_class, centroids, ('ro', 'bo'), stdDevs)

    x1 = np.vstack([p1[0], p1[1]])
    x2 = np.vstack([p2[0], p2[1]])

    labels1 = np.ones((1, len(p1[0])))
    labels2 = np.zeros((1, len(p2[0]))) - 1

    x, y = stackAndShuffleData([x1, x2], [labels1, labels2])
    x = np.vstack([x, np.ones(N_CLASS * 2)])
    return p1, p2, x, y[0]


if __name__ == "__main__":
    N_CLASS = 100
    p1, p2, x, y = generateStructureShuffleData(N_CLASS, ((1, 3), (7, 9)), (1, 1))

    net = getTrainedModel(x, y)
    getAccuracy(net, x, y)

    print(net.weights)
    w1, w2, b = net.weights
    print(getDecisionBoundry(w1, w2, b))
    plotPoints([p1, p2])
    xTest = np.array([[0, 0, 1], [8, 8, 1]]).T
    print(net.predictRaw(xTest))
    print(xTest)
