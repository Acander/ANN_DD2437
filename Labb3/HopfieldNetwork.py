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

    def _checkIfConverged(self, x, history):
        # print(history[-2:])
        for h in history:
            #    print(h, x, np.sum(np.abs(x - h)) == 0)
            if (np.sum(np.abs(x - h)) == 0):
                return True
        return False

    def predict(self, x):
        '''
        Expects a numpy array
        '''
        history = [x.copy()]
        # print("\n", x, "\n")
        epochCounter = 0
        while (True):
            epochCounter += 1
            # print(epochCounter)
            newX = []
            for i in range(len(x)):
                value = x * self.weights[i]
                newX.append(1 if np.sum(value) > 0 else -1)
                # print(value, np.sum(value), newX[-1] == x[i])
            x = np.array(newX)
            # print("")
            if (self._checkIfConverged(x, history)):
                return x, epochCounter, history + [x]

            history.append(x.copy())

    def sequentialPredict(self, x, numIteration=10000):
        '''
        Expects a numpy array
        '''
        history = [x.copy()]
        lastChanged = 0
        for epoch in range(numIteration):
            newX = x.copy()
            i = np.random.choice(range(len(x)))
            newX[i] = 1 if np.sum(x * self.weights[i]) > 0 else -1
            if (newX[i] != x[i]):
                lastChanged = epoch + 1

            x = np.array(newX)
            history.append(x.copy())

        return x, lastChanged, history


if __name__ == '__main__':
    patterns = [
        [-1, -1, 1, -1, 1, -1, -1, 1],
        [-1, -1, -1, -1, -1, 1, -1, -1],
        [-1, 1, 1, -1, -1, 1, -1, 1],
    ]
    noisyPatterns = [
        [1, -1, 1, -1, 1, -1, -1, 1],
        [1, 1, -1, -1, -1, 1, -1, -1],
        [1, 1, 1, -1, 1, 1, -1, 1],
    ]

    model = HopsNet(8)
    model.setWeights(patterns)
    # print(model.weights)

    for i, n in enumerate(noisyPatterns):
        final, epochs, history = model.sequentialPredict(np.array(n))
        print(np.sum(np.abs(final - patterns[i])), epochs)
    print("\n", final)
