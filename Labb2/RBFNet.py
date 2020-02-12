import numpy as np

from Labb2.RBFFunc import RBF


class RadialBasisFunctionNetwork:

    def __init__(self, inputSize, outSize, hiddenSize, hiddenCentroids, RBF, lr=0.0005, l1Dist=False):
        self.outWeights = np.random.normal(-0.05, 0.05, (hiddenSize, outSize))
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
        loss = (pred - labels)  # Allows for batch
        #loss = np.abs(pred - labels)  # Allows for batch
        #print("Previous:", self.previousHiddenValues, self.previousHiddenValues.shape)
        for i in range(pred.shape[1]):
            outputLoss = loss[:, i].reshape((loss.shape[0], 1))
            # print("Loss {}\n".format(i), outputLoss, outputLoss.shape)

            # self.previousHiddenValues = self.previousHiddenValues.reshape(1, self.previousHiddenValues.shape[0])
            weightGrad = outputLoss * self.previousHiddenValues

            # print("Grad:", weightGrad, weightGrad.shape)
            weightGrad = np.mean(weightGrad, axis=0)  # Batch Grad mean
            weightGrad = weightGrad.reshape(weightGrad.shape + (1,))  # Reshape for broadcast
            # print("Weights:", self.outWeights, self.outWeights.shape)
            # print("Grad:", weightGrad, weightGrad.shape)
            # print(self.lr, self.lr.shape)
            # print((self.lr * weightGrad).shape)
            self.outWeights[:, i] -= self.lr * weightGrad[:, 0]

            # input()
        return np.mean((pred - labels))

    def fit2(self, inputs, labels):
        pred = self.predict(inputs)
        loss = (labels - pred)

        weightGrads = [[] for i in range(self.outWeights.shape[0])]
        #print(pred.shape, loss.shape)
        for i in range(pred.shape[1]):  # Iterate over outputs
            outputLoss = loss[:, i]

            for j, l in enumerate(outputLoss):  # Iterate over samples in batch
                prevValues = self.previousHiddenValues[j]
                #print("OutLoss:", outputLoss.shape, "PrevValues:", prevValues.shape)
                for k, w in enumerate(self.outWeights):
                    wGrad = l * prevValues[k]
                    weightGrads[k].append(wGrad)

        for i, wGrad in enumerate(weightGrads):
            self.outWeights[i] += self.lr * np.mean(wGrad)

        return np.mean(np.abs(pred - labels))


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
    for i in range(1000000):
        loss = model.fit(inData, labels)
        print(loss)
        # print("Predict", model.predict(inData))
