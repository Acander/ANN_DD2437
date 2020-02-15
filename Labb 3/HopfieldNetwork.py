import numpy as np


class HopsNet:

    def __init__(self, numberOfNodes):
        self.weights = np.random.random((numberOfNodes, numberOfNodes))
        self.numNodes = numberOfNodes

    def setWeights(self, patterns, setDiagZero=False):
        '''
        Expects a nested list.
        Where every inner list is a pattern.
        '''
        data = np.array(patterns)
        results = []
        for p in data:
            p = p.reshape(p.shape + (1,))
            results.append(p.T * p)

        final = np.array(results)
        self.weights = np.sum(final, axis=0)
        if (setDiagZero):
            np.fill_diagonal(self.weights, 0)

    def _checkIfMinima(self, pattern):
        for i, p in enumerate(pattern):
            pass

    def sequentialPredict(self, pattern):
        hasChanged = True
        while (hasChanged):
            hasChanged = False

    def predict(self):

        pass


if __name__ == '__main__':
    patterns = [
        [-1, -1, 1, -1, 1, -1, -1, 1],
        [-1, -1, -1, -1, -1, 1, -1, -1],
        [-1, 1, 1, -1, -1, 1, -1, 1],
    ]
    model = HopsNet(8)
    model.setWeights(patterns)
    print(model.weights)
