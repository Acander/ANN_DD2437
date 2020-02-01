import numpy as np


class RadialBasisFunctionNetwork:

    def __init__(self, inputSize, outSize, hiddenSize, hiddenCentroids, RBF, l1Dist=False):
        self.outWeights = np.random.normal(0, 0.5, (hiddenSize, outSize))
        self.centroids = hiddenCentroids
        self.numHidden = hiddenSize
        self.l1Dist = l1Dist
        self.RBF = np.vectorize(RBF)
        assert hiddenCentroids.shape[0] == hiddenSize and hiddenCentroids.shape[1] == inputSize

    def predictDist(self, inputs):
        inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1])  # format input data for efficent broadcasting
        inputs = inputs.repeat(self.numHidden, axis=1)

        deltaDist = inputs - self.centroids
        if (self.l1Dist == False):  # L2 dist
            deltaDist **= 2
            dist = np.sum(deltaDist, axis=-1)
            return np.sqrt(dist)

        return np.sum(deltaDist, axis=-1)  # L1 Dist

    def _calcRBF(self, distances):
        return self.RBF(distances)

    def predict(self, inputs):
        nodeDists = self.predictDist(inputs)
        nodeScores = self._calcRBF(nodeDists)
        outputs = np.matmul(nodeScores, self.outWeights)
        return outputs



    def fit(self, inputs, labels):
        x = 10


if __name__ == '__main__':
    RBF = lambda x: x
    model = RadialBasisFunctionNetwork(2, 1, 4, np.array([[0, 0], [3, 3], [1, 1], [2, 2]]), RBF, l1Dist=False)
    print(model.predict(np.array([[2, 2], [3, 7]])))
