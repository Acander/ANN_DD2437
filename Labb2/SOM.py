import numpy as np

from Labb2.CL import closestCentroid
from Labb2.DataHandler import plotPointsXY
from Labb2.DataSetHandler import importAnimalDataSet

STEP_SIZE = 0.2
SIGMA0 = 1
TAU = 100


def distToWinnerChain(winner, nodes):
    # distances = []
    # for i in range(len(nodes)):
    #     distances.append(np.abs(winner - i))
    return np.abs(winner - np.arange(len(nodes)))


def distToWinnerCircular(winner, nodes):
    n = len(nodes)
    indices = np.arange(n)
    # print(winner)
    # print(nodes)
    vec1 = np.abs(winner - indices)
    vec2 = np.array(n - indices + 1 + winner)
    '''
    print("vec1:", vec1)
    print("vec2:", vec2)
    print(len(vec1))
    print(len(vec2))
    print(vec1[0])
    print(vec2[0])
    print(min(vec1[0], vec2[0]))
    '''
    return np.array([min(vec1[i], vec2[i]) for i in range(len(vec1))])
    # return np.min(
    #     np.abs(winner - indices),
    #     n - indices + 1 + winner)


def neighbFunc(sigma, dist):
    return np.exp(- (dist * dist) / (2 * sigma * sigma))


def meanLength(W):
    return np.mean(np.linalg.norm(W, axis=1))


def SOM(X, inSize, outSize, epochs=20, distFunc=distToWinnerChain, sort=True):
    numCategories = len(X)
    W = np.random.uniform(0, 1, (outSize, inSize))  # These are the nodes, each row defines a centroid

    for t in range(epochs):
        sigma = SIGMA0 * np.exp(-(t * t) / TAU)
        for i in range(numCategories):
            winner = closestCentroid(X[i], W)
            distances = distFunc(winner, W)
            h = neighbFunc(sigma, distances)
            # print(np.shape(h))
            # print(np.shape(X[i]))
            for wIdx in range(len(W)):
                # print(np.shape(W[wIdx]))
                # print(np.shape(X[i] - W[wIdx]))
                # print(np.linalg.norm(X[i] - W[wIdx]))
                W[wIdx] += STEP_SIZE * h[wIdx] * (X[i] - W[wIdx])

    # print(meanLength(W))
    winners = []
    for i in range(numCategories):
        winners.append((i, closestCentroid(X[i], W)))
    return sorted(winners, key=lambda x: x[1]) if sort else winners


def somAnimals():
    X, animalNames = importAnimalDataSet()
    similaritySequence = SOM(X, 84, 100, epochs=100, distFunc=distToWinnerChain)
    animalsOrdered = [animalNames[animal[0]] for animal in similaritySequence]
    print(animalsOrdered)


def somCyclicTour():
    X = np.random.uniform(0.0, 1.0, (10, 2))
    print(X)
    similaritySequence = SOM(X, 2, 10, epochs=20, distFunc=distToWinnerCircular, sort=False)
    print(similaritySequence)
    xPlot = [X[i][0] for i, winner in similaritySequence]
    yPlot = [X[i][1] for i, winner in similaritySequence]
    xPlot.append(X[0][0])
    yPlot.append(X[1][0])
    # end not included? no cycle in plot
    print([(xPlot, yPlot)])
    plotPointsXY([(xPlot, yPlot)], ["lol"], True)


if __name__ == '__main__':
    somCyclicTour()
    # somAnimals()
