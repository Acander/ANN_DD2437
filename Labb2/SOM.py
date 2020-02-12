import numpy as np

from Labb2.CL import closestCentroid
from Labb2.DataHandler import plotPointsXY
from Labb2.DataSetHandler import importAnimalDataSet, importCities, importVotesAndMPs

STEP_SIZE = 0.2
# SIGMA0 = 1
SIGMA0 = 10
TAU = 2000


def distToWinnerChain(winner, nodes):
    return np.abs(winner - np.arange(len(nodes)))


def distToWinnerCircular(winner, nodes):
    n = len(nodes)
    indices = np.arange(n)
    vec1 = np.abs(winner - indices)
    vec2 = np.array(n - indices + 1 + winner)
    return np.array([min(vec1[i], vec2[i]) for i in range(len(vec1))])


def distToWinner2DGrid(winner, nodes):
    # numNodes = len(nodes)
    # n = int(np.sqrt(numNodes))
    numNodes = 100
    n = 10
    winnerY = int(winner / n)
    winnerX = winner % n
    # print(winnerY, winnerX)
    # grid = np.reshape(nodes, (n, n, 2))

    distances = []
    for i in range(numNodes):
        currY = int(i / n)
        currX = i % n
        distances.append(np.abs(currY - winnerY) + np.abs(currX - winnerX))

    return np.array(distances)


def neighbFunc(sigma, dist):
    res = np.exp(- (dist * dist) / (2 * sigma * sigma))
    # print(np.mean(res) - 0.01)
    return res


def meanLength(W):
    return np.mean(np.linalg.norm(W, axis=1))


def SOM(X, inSize, outSize, epochs=20, distFunc=distToWinnerChain, sort=True):
    numCategories = len(X)
    W = np.random.uniform(0, 1, (outSize, inSize))  # These are the nodes, each row defines a centroid

    for t in range(epochs):
        sigma = SIGMA0 * np.exp(-(t * t) / TAU)
        global STEP_SIZE
        # if t+1 % 100 == 0:
        # STEP_SIZE -= 0.1
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
    return W, sorted(winners, key=lambda x: x[1]) if sort else winners


def somAnimals():
    X, animalNames = importAnimalDataSet()
    W, similaritySequence = SOM(X, 84, 100, epochs=100, distFunc=distToWinnerChain)
    animalsOrdered = [animalNames[animal[0]] for animal in similaritySequence]
    print(animalsOrdered)


def somCyclicTour():
    X = importCities()
    np.random.shuffle(X)
    W, similaritySequence = SOM(X, 2, 10, epochs=300, distFunc=distToWinnerCircular, sort=True)
    # print(similaritySequence)
    xPlot = [X[i][0] for i, winner in similaritySequence]
    yPlot = [X[i][1] for i, winner in similaritySequence]
    firstIdx = similaritySequence[0][0]
    # Include first point again, to show the cycle
    xPlot.append(X[firstIdx][0])
    yPlot.append(X[firstIdx][1])

    weightsX = W[:, 0]
    weightsY = W[:, 1]
    weightsX = np.concatenate((weightsX, [W[0][0]]))
    weightsY = np.concatenate((weightsY, [W[0][1]]))
    plotPointsXY([(xPlot, yPlot), (weightsX, weightsY)], ["OrderedCitiesSOM", "Weights SOM"], True)

    # plotPointsXY([(weightsX, weightsY)], ["Route"], True)


def unpackVoteData(finalMPInfoList):
    names = []
    genders = []
    districts = []
    partyIdxs = []
    partyNames = []
    partyColors = []
    for name, gender, district, partyTuple in finalMPInfoList:
        names.append(name)
        genders.append(gender)
        districts.append(district)
        partyIndex, partyName, partyColor = partyTuple
        partyIdxs.append(partyIndex)
        partyNames.append(partyName)
        partyColors.append(partyColor)

    return names, genders, districts, partyIdxs, partyNames, partyColors


# similaritySequence = [ (mpIndex, nodeWinnerIdx) ...]
def plotVotes(similaritySequence, partyColors, genders, districts):
    n = 10
    mpIdx = [idx for idx, _ in similaritySequence]
    # print(mpIdx)
    winnerIdx = [winner for _, winner in similaritySequence]
    colorsSorted = [partyColors[idx] for idx in mpIdx]
    pointsX = [i % n + np.random.uniform(-0.25, 0.25) for i in winnerIdx]
    pointsY = [int(i / n) + np.random.uniform(-0.25, 0.25) for i in winnerIdx]
    shapes = ["x" if genders[idx] == "Male" else "o" for idx, _ in similaritySequence]
    districtsSorted = [districts[idx] for idx, _ in similaritySequence]
    plotPointsXY([(pointsX, pointsY)], [""], drawPoints=True, drawLines=False, colors=colorsSorted, shape=shapes,
                 districts=districtsSorted)


def somVotes():
    np.random.seed(1337)
    votes, finalMPInfoList = importVotesAndMPs()
    # np.random.shuffle(votes)
    names, genders, districts, partyIdxs, partyNames, partyColors = unpackVoteData(finalMPInfoList)
    numEpochs = 30
    global TAU
    TAU *= (numEpochs/200)

    # WPrev, similaritySequencePrev = SOM(votes, 31, 100, epochs=0, distFunc=distToWinner2DGrid, sort=True)
    W, similaritySequence = SOM(votes, 31, 100, epochs=numEpochs, distFunc=distToWinner2DGrid, sort=True)
    # newSimSeq = [(similaritySequence[i][0], similaritySequence[i][1], partyColors[similaritySequence[i][0]])
    #              for i in range(len(similaritySequence))]
    # plotVotes(similaritySequencePrev, partyColors, genders, districts)
    plotVotes(similaritySequence, partyColors, genders, districts)


if __name__ == '__main__':
    # somCyclicTour()
    # somAnimals()
    somVotes()
