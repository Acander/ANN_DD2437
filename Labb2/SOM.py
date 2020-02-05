import numpy as np

from Labb2.CL import closestCentroid

STEP_SIZE = 0.2
SIGMA0 = 1
TAU = 1


def distToWinner(winner, nodes):
    # distances = []
    # for i in range(len(nodes)):
    #     distances.append(np.abs(winner - i))
    return np.abs(winner - np.arange(len(nodes)))


def neighbFunc(sigma, dist):
    return np.exp(- (dist * dist) / (2 * sigma * sigma))


def SOM1d(X, inSize, outSize, epochs=20):
    numCategories = len(X)
    W = np.random.uniform(0, 1, (outSize, inSize))  # These are the nodes, each row defines a centroid

    for t in range(epochs):
        sigma = SIGMA0 * np.exp(-(t * t) / TAU)
        for i in range(numCategories):
            winner = closestCentroid(X[i], W)
            distances = distToWinner(winner, W)
            h = neighbFunc(sigma, distances)
            # print(np.shape(h))
            # print(np.shape(X[i]))
            for wIdx in range(len(W)):
                # print(np.shape(W[wIdx]))
                # print(np.shape(X[i] - W[wIdx]))
                # print(np.linalg.norm(X[i] - W[wIdx]))
                W[wIdx] += STEP_SIZE * h[wIdx] * np.linalg.norm(X[i] - W[wIdx])

    winners = []
    for i in range(numCategories):
        winners.append(closestCentroid(X[i], W))
    print(np.sort(winners))


if __name__ == '__main__':
    X = 1337
    SOM1d(np.random.normal(1, 1, (32, 84)), 84, 100)
