import numpy as np
import numpy.random as r
import matplotlib.pyplot as plt


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


def plotPointsXY(pointsList, labels, drawPoints=False):
    for i, points in enumerate(pointsList):
        print(points)
        x, y = points
        plt.plot(x, y, label=labels[i])
        if drawPoints:
            plt.plot(x, y, "go")
    plt.legend()
    plt.show()


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
