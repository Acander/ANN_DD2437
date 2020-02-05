import numpy as np

from Labb2.CL import closestCentroid
from Labb2.DataSetHandler import importAnimalDataSet

STEP_SIZE = 0.2
SIGMA0 = 1
TAU = 100


def distToWinner(winner, nodes):
    # distances = []
    # for i in range(len(nodes)):
    #     distances.append(np.abs(winner - i))
    return np.abs(winner - np.arange(len(nodes)))


def neighbFunc(sigma, dist):
    # print(sigma)
    return np.exp(- (dist * dist) / (2 * sigma * sigma))


def meanLength(W):
    # distSum = 0
    # for i in range(len(W)):
    #     distSum += np.linalg.norm(W[i])
    return np.mean(np.linalg.norm(W, axis=1))


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
                W[wIdx] += STEP_SIZE * h[wIdx] * (X[i] - W[wIdx])

    print(meanLength(W))
    winners = []
    for i in range(numCategories):
        winners.append((i, closestCentroid(X[i], W)))
    return sorted(winners, key=lambda x: x[1])


if __name__ == '__main__':
    X, animalNames = importAnimalDataSet()
    animalNames = [animal.strip("\t\'") for animal in animalNames]
    # SOM1d(np.random.normal(1, 1, (32, 84)), 84, 100)
    similaritySequence = SOM1d(X, 84, 100, epochs=20)
    animalsOrdered = [animalNames[animal[0]] for animal in similaritySequence]
    print(animalsOrdered)
