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
    pattern = np.reshape(pattern, (1, len(pattern)))
    return - np.sum(weights * (pattern.T @ pattern))

    # energySum = 0
    # for i in range(len(pattern)):
    #     for j in range(len(pattern)):
    #         energySum += weights[i][j] * pattern[i] * pattern[j]
    # return -energySum


def existsInList(listOfArrs, arr):
    for a in listOfArrs:
        if np.array_equal(arr, a):
            return True
    return False


if __name__ == '__main__':
    # print(allPossibleSequences(8))
    seq = np.array([-1, -1, 1])
    newSeq = flipBits(seq, 3)
    print(seq)
    print(newSeq)
