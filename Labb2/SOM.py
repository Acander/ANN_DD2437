import numpy as np

from Labb2.CL import closestCentroid
from Labb2.DataHandler import plotPointsXY
from Labb2.DataSetHandler import importAnimalDataSet, importCities

STEP_SIZE = 0.2
# SIGMA0 = 1
SIGMA0 = 3
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
            for wIdx in range(len(W)):
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
    '''
    X = np.random.uniform(0.0, 1.0, (10, 2))
    X = [[0.36774487, 0.97798252],
         [0.7865123, 0.26115321],
         [0.13692526, 0.87530038],
         [0.41689718, 0.02890185],
         [0.95716118, 0.61338133],
         [0.00175163, 0.03152998],
         [0.044273, 0.9803899],
         [0.76049118, 0.09521741],
         [0.22803144, 0.98443609],
         [0.794949, 0.50350659]]
    '''
    X = importCities()
    np.random.shuffle(X)
    similaritySequence = SOM(X, 2, 10, epochs=50, distFunc=distToWinnerCircular, sort=True)
    # print(similaritySequence)
    xPlot = [X[i][0] for i, winner in similaritySequence]
    yPlot = [X[i][1] for i, winner in similaritySequence]
    firstIdx = similaritySequence[0][0]
    # Include first point again, to show the cycle
    xPlot.append(X[firstIdx][0])
    yPlot.append(X[firstIdx][1])
    plotPointsXY([(xPlot, yPlot)], ["Route"], True)


if __name__ == '__main__':
    somCyclicTour()
    # somAnimals()
