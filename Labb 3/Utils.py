import numpy as np


def allPossibleSequences(length):
    seqs = np.empty((2 ** length, length))
    for i in range(2 ** length):
        binaryString = ('{0:0' + str(length) + 'b}').format(i)
        for j in range(length):
            seqs[i][j] = (int(binaryString[j]) * 2) - 1

    return seqs


if __name__ == '__main__':
    print(allPossibleSequences(8))
