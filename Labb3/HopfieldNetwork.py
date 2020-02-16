import numpy as np


class HopsNet:

    def __init__(self, numberOfNodes):
        self.weights = np.random.random((numberOfNodes, numberOfNodes))
        self.numNodes = numberOfNodes
        self.removeBias = False

    def setWeights(self, patterns, setDiagZero=False, removeBias=False):
        '''
        Expects a nested list.
        Where every inner list is a pattern.
        '''
        # patterns = np.dtype(np.copy(patterns), np.float64)

        if removeBias:
            self.removeBias = True
            avgActivity = np.mean(patterns)
            patterns = patterns - avgActivity

        data = np.array(patterns)
        results = []
        for p in data:
            p = p.reshape(p.shape + (1,))
            results.append(p.T * p)

        final = np.array(results)
        self.weights = np.sum(final, axis=0)
        # self.weights = np.average(final, axis=0)
        if (setDiagZero):
            np.fill_diagonal(self.weights, 0)

    def _checkIfConverged(self, x, history):
        # print(history[-2:])
        for h in history:
            #    print(h, x, np.sum(np.abs(x - h)) == 0)
            if (np.sum(np.abs(x - h)) == 0):
                return True
        return False

    def predict(self, x, theta=0):
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
                if self.removeBias:
                    newX.append(0.5 + 0.5 * np.sign(np.sum(x * self.weights[i]) - theta))
                else:
                    value = x * self.weights[i]
                    newX.append(np.sign(np.sum(value)))
                # print(value, np.sum(value), newX[-1] == x[i])
            x = np.array(newX)
            # print("")
            if (self._checkIfConverged(x, history)):
                return x, epochCounter, history + [x]

            history.append(x.copy())

    def sequentialPredict(self, x, numIteration=10000, theta=0):
        '''
        Expects a numpy array
        '''
        history = [x.copy()]
        lastChanged = 0
        for epoch in range(numIteration):
            newX = x.copy()
            i = np.random.choice(range(len(x)))
            if self.removeBias:
                newX[i] = 0.5 + 0.5 * np.sign(np.sum(x * self.weights[i]) - theta)
            else:
                newX[i] = np.sign(np.sum(x * self.weights[i]))
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
