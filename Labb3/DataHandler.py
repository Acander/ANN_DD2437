import numpy as np
import matplotlib.pyplot as plt

'''
def importPatternData():
    x1D = [1, -1, 1, -1, 1, -1, -1, 1]
    x2D = [1, 1, -1, -1, -1, 1, -1, -1]
    x3D = [1, 1, 1, 1, -1, 1, 1, -1, 1]
    return x1D, x2D, x3D
'''


def importAllPicData():
    picDatas = open("pict.dat", 'r').read()
    picDatas = np.array(list(map(int, picDatas.split(","))))
    picDatas = picDatas.reshape((11, 1024))
    return picDatas  # Returns a matrix where each row represents a picture (unformated)


'''
def extractPicSet(picIndex):
    pics = importAllPicData()
    pic = pics[picIndex - 1]
    return pic  # Returns an array representing a certain pic (unformated)


def formatPic(pic):
    return pic.reshape((32, 32))


def plotPic(picIndex):
    pic = formatPic(extractPicSet(picIndex - 1))
    for i in range(32):
        for j in range(32):
            if pic[i][j] == 1:
                plt.scatter(i, j, color="black")
            else:
                plt.scatter(i, j, color="white")
    plt.show()
'''


def plotDatapoint(dataPoint):
    xyLen = int(np.sqrt(len(dataPoint)))
    pic = np.reshape(dataPoint, (xyLen, xyLen))
    for i in range(xyLen):
        for j in range(xyLen):
            if pic[i][j] == 1:
                plt.scatter(i, j, color="black")
            else:
                plt.scatter(i, j, color="white")
    plt.show()


def generateSparsePattern(N, activityRatio, numPatterns):
    patterns = np.zeros((numPatterns, N))
    idxRange = np.arange(N)
    numOnes = int(N * activityRatio)
    for i in range(numPatterns):
        indices = np.random.choice(idxRange, numOnes, replace=False)
        patterns[i][indices] = 1

    return patterns


if __name__ == '__main__':
    # importPicData()
    # extractPicSet(5)
    # plotPic(3)
    data = importAllPicData()
    data = (data + 1) / 2
    for dp in data:
        print(np.mean(dp))

    # p = generateSparsePattern(10, 0.7, 5)
    # print(p)
