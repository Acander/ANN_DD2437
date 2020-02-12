import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def generateData(box=False, shuffle=True, noiseVariance=0.0):
    xTrain = np.arange(0, 2 * np.pi, 0.1)
    xTest = np.arange(0.05, 2 * np.pi, 0.1)
    if shuffle:
        r.shuffle(xTrain)
        r.shuffle(xTest)

    yTrain = np.sin(xTrain * 2)
    yTest = np.sin(xTest * 2)

    if box:
        yTrain[np.where(yTrain >= 0)] = 1
        yTrain[np.where(yTrain < 0)] = -1
        yTest[np.where(yTest >= 0)] = 1
        yTest[np.where(yTest < 0)] = -1

    if noiseVariance > 0:
        yTrain += np.random.normal(0, np.sqrt(noiseVariance), len(xTrain))
        yTest += np.random.normal(0, np.sqrt(noiseVariance), len(xTest))

    N = len(xTrain)

    return np.reshape(xTrain, (N, 1)), \
           np.reshape(yTrain, (N, 1)), \
           np.reshape(xTest, (N, 1)), \
           np.reshape(yTest, (N, 1))


def evaluateModel(model, X, Y, residualError=True):
    predictions = model.predict(X)
    error = np.abs(Y - predictions) if residualError else np.square(Y - predictions)
    return np.mean(error)


def plotBars(districts, leftRightValues):
    meanRightValuePerDistrict = []
    for i in range(1, 29 + 1):
        indicesCurrentDistrict = np.where(np.array(districts) == i)
        # print("i =", i, ": ", indicesCurrentDistrict)
        meanRight = np.mean(leftRightValues[indicesCurrentDistrict])
        meanRightValuePerDistrict.append(meanRight)

    districtList = [str(i) for i in range(1, 29 + 1)]
    plt.bar(districtList, meanRightValuePerDistrict)
    plt.xlabel("District")
    plt.ylabel("Average first feature (left/right scale)")
    plt.show()


def plotPointsXY(pointsList, labels, drawPoints=False, drawLines=True, colors=None, shape=None, districts=None):
    S = mpatches.Patch(color='red', label='S')
    M = mpatches.Patch(color='blue', label='M')
    FP = mpatches.Patch(color='aqua', label='FP')
    V = mpatches.Patch(color='purple', label='V')
    MP = mpatches.Patch(color='green', label='MP')
    KD = mpatches.Patch(color='teal', label='KD')
    C = mpatches.Patch(color='lime', label='C')
    No = mpatches.Patch(color='white', label='None')

    for i, points in enumerate(pointsList):
        x, y = points
        if drawLines:
            plt.plot(x, y, label=labels[i])
        if drawPoints:
            if colors is None:
                plt.plot(x, y, "bo" if i == 0 else "ro")
            else:
                for pIdx in range(len(x)):
                    plt.scatter(10 - y[pIdx], x[pIdx], color=colors[pIdx], linewidths=1, edgecolors="black",
                                marker=shape[pIdx])
                    # plt.annotate(districts[pIdx], (x[pIdx], y[pIdx]), textcoords="offset points", xytext=(10, -4), ha='center')
    plt.legend(handles=[S, M, FP, V, MP, KD, C, No])
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    plotBars(districts, 10 - np.array(pointsList[0][1]))


def plotPoints(pointsList, colors, labels, sizes=None, twoD=False):
    if sizes is None:
        sizes = [5] * len(pointsList)

    for i in range(len(pointsList)):
        if not twoD:
            plt.plot(pointsList[i], np.zeros(len(pointsList[i])), colors[i], label=labels[i], markersize=sizes[i])
        else:
            plt.plot(pointsList[i], colors[i], label=labels[i], markersize=sizes[i])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    xTrain, yTrain, xTest, yTest = generateData(False, noiseVariance=0.0)
