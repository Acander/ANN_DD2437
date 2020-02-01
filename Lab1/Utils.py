import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D


def plotPoints(points, line=None, line2=None, label1="", label2=""):
    '''
    Expects points to be a list of tuples:
        Tuple1: (xPoints, yPoints, color)
    '''
    plt.xlabel('x')
    plt.ylabel('y')

    labels = ['50% pruned', '80%/20% pruned']
    labelsVal = ['50% pruned Validation', '80%/20% pruned Validation']
    i = 0
    for x, y, color in points:
        label = labels[int(i/2)] if i % 2 == 0 else labelsVal[int(i/2)]
        plt.plot(x, y, color, markeredgecolor='white', label=label)
        i += 1

    xMin = np.min(np.concatenate(([p[0] for p in points]))) - 1
    xMax = np.max(np.concatenate(([p[0] for p in points]))) + 1

    yMin = np.min(np.concatenate(([p[1] for p in points]))) - 1
    yMax = np.max(np.concatenate(([p[1] for p in points]))) + 1

    if line is not None:
        s, b = line
        plotLinearDecisionBoundry(s, b, xMin - 10, xMax + 10, color="green", label=label1)

    if line2 is not None:
        s, b = line2
        plotLinearDecisionBoundry(s, b, xMin - 10, xMax + 10, color="purple", label=label2)

    setPlotDim([xMin, xMax], [yMin, yMax])
    plt.legend()
    plt.show()


def setPlotDim(xAxis, yAxis, zAxis=None):
    plt.xlim(xAxis[0], xAxis[1])
    plt.ylim(yAxis[0], yAxis[1])


def plot3D(points):
    print(points)

    fig = plt.figure()
    ax = Axes3D(fig)
    for x, y, z, color in points:
        ax.scatter(x, y, z, color=color)

    xMin = np.min(np.concatenate(([p[0] for p in points]))) - 1
    xMax = np.max(np.concatenate(([p[0] for p in points]))) + 1

    yMin = np.min(np.concatenate(([p[1] for p in points]))) - 1
    yMax = np.max(np.concatenate(([p[1] for p in points]))) + 1

    zMin = np.min(np.concatenate(([p[2] for p in points]))) - 1
    zMax = np.max(np.concatenate(([p[2] for p in points]))) + 1

    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    ax.set_zlim(zMin, zMax)

    plt.show()


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


def plotDecisionBoundary(model, points):
    h = 0.02  # step size in the mesh
    xRange = np.arange(-2, 2, h)
    yRange = np.arange(-2, 2, h)
    xx, yy = np.meshgrid(xRange, yRange)
    x = np.c_[xx.ravel(), yy.ravel()].T

    Z = model.forwardPass(x)
    Z.reshape(xx.shape)
    Z = np.reshape(Z, (len(xRange), len(yRange)))
    plt.contourf(xx, yy, Z, cmap='coolwarm')
    plotPoints(points)
    # plt.show()
