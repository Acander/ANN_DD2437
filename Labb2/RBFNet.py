import numpy as np

from Labb2.RBFFunc import RBF


class RadialBasisFunctionNetwork:

    def __init__(self, inputSize, outSize, hiddenSize, hiddenCentroids, RBF, lr=0.0001, l1Dist=False):
        self.outWeights = np.random.normal(0, 0.5, (hiddenSize, outSize))
        self.centroids = hiddenCentroids
        self.numHidden = hiddenSize
        self.l1Dist = l1Dist
        self.RBF = np.vectorize(RBF)
        self.previousHiddenValues = "CACHED FROM PREDICTION"
        self.lr = np.array([lr])
        assert hiddenCentroids.shape[0] == hiddenSize and hiddenCentroids.shape[1] == inputSize

    def predictDist(self, inputs):
        inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1])  # format input data for efficent broadcasting
        inputs = inputs.repeat(self.numHidden, axis=1)

        deltaDist = inputs - self.centroids
        if (self.l1Dist == False):  # L2 dist
            deltaDist **= 2
            return np.sum(deltaDist, axis=-1)

        return np.sum(deltaDist, axis=-1)  # L1 Dist

    def predict(self, inputs):
        nodeDists = self.predictDist(inputs)
        # print(nodeDists)
        self.previousHiddenValues = self.RBF(nodeDists)
        # print(self.previousHiddenValues)
        return np.matmul(self.previousHiddenValues, self.outWeights)

    def fit(self, inputs, labels):
        pred = self.predict(inputs)
        for i in range(pred.shape[1]):
            # print(pred.shape, labels.shape)
            # print("Pred\n", pred)
            # print((pred - labels)**2)
            loss = (pred - labels) ** 2  # Allows for batch
            # print("Loss\n", loss, loss.shape)
            # loss = loss.reshape(loss.shape + (1,))
            # print(loss, loss.shape)
            self.previousHiddenValues = self.previousHiddenValues

            # print("Previous:", self.previousHiddenValues, self.previousHiddenValues.shape)
            # self.previousHiddenValues = self.previousHiddenValues.reshape(1, self.previousHiddenValues.shape[0])
            weightGrad = loss[:, i].reshape((loss.shape[0], 1)) * self.previousHiddenValues
            # print("Grad:", weightGrad, weightGrad.shape)
            weightGrad = np.mean(weightGrad, axis=0)  # Batch Grad mean
            weightGrad = weightGrad.reshape(weightGrad.shape + (1,))  # Reshape for broadcast
            # print("Weights:", self.outWeights, self.outWeights.shape)
            # print("Grad:", weightGrad, weightGrad.shape)
            # print(self.lr, self.lr.shape)
            # print((self.lr * weightGrad).shape)
            self.outWeights[:, i] += self.lr * weightGrad[:, 0]

            # input()
        return np.mean(loss)


if __name__ == '__main__':
    early = np.array([[0, 8], [3, 3], [1, 1], [2, 2]])
    centroids = np.random.uniform(-1, 7, (4, 2))
    print(centroids, centroids.shape)
    print(early.shape)
    model = RadialBasisFunctionNetwork(2, 2, 4, centroids, RBF, l1Dist=False)
    # print(model.predict(np.array([[2, 2], [3, 7]])))
    # [(points), () ]
    inData = np.array([[2, 2], [0, 0], [3, 3]])
    labels = np.array([[1, 3], [-1, 1], [5, 5]])
    for i in range(100000):
        loss = model.fit(inData, labels)
        print(loss)
        #print("Predict", model.predict(inData))
