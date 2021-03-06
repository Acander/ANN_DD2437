import numpy as np


def allPossibleSequences(length):
    seqs = np.empty((2 ** length, length))
    for i in range(2 ** length):
        binaryString = ('{0:0' + str(length) + 'b}').format(i)
        for j in range(length):
            seqs[i][j] = (int(binaryString[j]) * 2) - 1

    return seqs


def flipBits(pattern, numBits):
    copy = np.copy(pattern)
    indices = np.random.choice(np.arange(len(pattern), dtype=int), numBits, replace=False)
    # print(indices)
    copy[indices] *= -1
    return copy


def energy(weights, pattern):
    pattern = np.expand_dims(pattern, axis=0)  # pattern.reshape(1, len(pattern))
    return - np.sum(weights * (pattern.T @ pattern))

    # energySum = 0
    # for i in range(len(pattern)):
    #     for j in range(len(pattern)):
    #         energySum += weights[i][j] * pattern[i] * pattern[j]
    # return -energySum


def generateRandomWeightMatrix(numberOfNodes, symmetric=False):
    w = np.random.normal(0, 1, (numberOfNodes, numberOfNodes))
    if (symmetric == False):
        return w
    return (w + w.T) / 2
