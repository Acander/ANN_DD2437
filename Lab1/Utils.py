import matplotlib.pyplot as plt
import numpy as np


def plotPoints(points, line=None, line2=None, label1="bsda", label2="asda"):
    '''
    Expects points to be a list of tuples:
        Tuple1: (xPoints, yPoints, color)
    '''
    plt.xlabel('x')
    plt.ylabel('y')

    for x, y, color in points:
        plt.plot(x, y, color)

    xMin = np.min(np.concatenate(([p[0] for p in points]))) - 1
    xMax = np.max(np.concatenate(([p[0] for p in points]))) + 1

    yMin = np.min(np.concatenate(([p[1] for p in points]))) - 1
    yMax = np.max(np.concatenate(([p[1] for p in points]))) + 1

    if line is not None:
        s, b = line
        plotLinearDecisionBoundry(s, b, xMin-10, xMax+10, color="green", label=label1)

    if line2 is not None:
        s, b = line2
        plotLinearDecisionBoundry(s, b, xMin-10, xMax+10, color="purple", label=label2)

    setPlotDim([xMin, xMax], [yMin, yMax])
    plt.legend()
    plt.show()


def setPlotDim(xAxis, yAxis):
    plt.xlim(xAxis[0], xAxis[1])
    plt.ylim(yAxis[0], yAxis[1])


def generate2DNormal(numberOfPoints, origo, std):
    return np.random.normal(0, std, (numberOfPoints, 2)) + origo


def generate2DNormalInCoords(numberOfPoints, origo, std):
    points = np.random.normal(0, std, (numberOfPoints, 2)) + origo
    return points[:, 0], points[:, 1]


def stackAndShuffleData(inData, labels):
    inData = np.vstack([d.T for d in inData])
    labels = np.vstack([l.T for l in labels])

    randomOrder = np.random.choice(range(len(inData)), len(inData), replace=False)
    inData = np.array([inData[i] for i in randomOrder]).T
    labels = np.array([labels[i] for i in randomOrder]).T
    return inData, labels


def generateData(pointsPerClass, centroids, colors, deviations):
    points = []
    for origo, color, std in zip(centroids, colors, deviations):
        x, y = generate2DNormalInCoords(pointsPerClass, origo, std)
        points.append((x, y, color))

    return points


def plotLearningCurves(metrics):
    pass


def plotLinearDecisionBoundry(slope, bias, min, max, color="green", label=""):
    x = np.linspace(min, max, 100)
    y = slope * x + bias
    plt.plot(x, y, '-r', color=color, label=label)
