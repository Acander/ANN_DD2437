import numpy as np
import matplotlib.pyplot as plt


def importPatternData():
    x1D = [1, -1, 1, -1, 1, -1, -1, 1]
    x2D = [1, 1, -1, -1, -1, 1, -1, -1]
    x3D = [1, 1, 1, 1, -1, 1, 1, -1, 1]
    return x1D, x2D, x3D


def importAllPicData():
    picDatas = open("pict.dat", 'r').read()
    picDatas = np.array(list(map(int, picDatas.split(","))))
    picDatas = picDatas.reshape((11, 1024))
    return picDatas  # Returns a matrix where each row represents a picture (unformated)


def extractPicSet(picIndex):
    pics = importAllPicData()
    pic = pics[picIndex-1]
    return pic  # Returns an array representing a certain pic (unformated)


def formatPic(pic):
    return pic.reshape((32, 32))


def plotPic(picIndex):
    pic = formatPic(extractPicSet(picIndex-1))
    for i in range(32):
        for j in range(32):
            if pic[i][j] == 1:
                plt.scatter(i, j, color="black")
            else:
                plt.scatter(i, j, color="white")
    plt.show()


if __name__ == '__main__':
    # importPicData()
    # extractPicSet(5)
    plotPic(3)
