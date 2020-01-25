import numpy as np
from Lab1.LinearSeparableData import generateData

# from Lab1.Utils import generate2DNormal, plotPoints
from Lab1.Utils import plotPoints


class PLNet:

    def __init__(self, learningRate, sizeIn):
        self.lr = learningRate
        self.weights = np.random.normal(0, 0.5, sizeIn + 1)

    def predict(self, x):
        # self.weights @ np.vstack([x, ([1] * np.size(x[0]))])
        return ((self.weights @ x > 0).astype(int) * 2) - 1  # convert to -1/1

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


if __name__ == "__main__":
    p1, p2 = generateData(10, ((2, 3), (5, 5)), ('ro', 'bo'), (1, 1))

    x1 = np.vstack([p1[0], p1[1]])
    x2 = np.vstack([p2[0], p2[1]])

    labels1 = np.ones((1, len(p1[0])))
    labels2 = np.zeros((1, len(p2[0])))

    plotPoints(p1)
