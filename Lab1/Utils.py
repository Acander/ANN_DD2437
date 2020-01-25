import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D



def plotPoints(points, epochs, batchSize, pointsPerClass, line=None):
    '''
    Expects points to be a list of tuples:
        Tuple1: (xPoints, yPoints, color)
    '''
    plt.xlabel('x1', color='black')
    plt.ylabel('x2', color='black')

    subtitle = "epochs: " + str(epochs) + ", batch size: " + str(batchSize) + ", points per class: " + str(pointsPerClass)
    plt.title("Delta Rule\n" + subtitle)
    #plt.suptitle("Delta Rule", fontsize=18)

    #plt.legend(loc="upper left", )

    for x, y, color in points:
        plt.plot(x, y, color)

    xMin = np.min(np.concatenate(([p[0] for p in points]))) - 1
    xMax = np.max(np.concatenate(([p[0] for p in points]))) + 1

    yMin = np.min(np.concatenate(([p[1] for p in points]))) - 1
    yMax = np.max(np.concatenate(([p[1] for p in points]))) + 1

    if line != None:
        s, b = line
        plotLinearDecisionBoundry(s, b, xMin - 10, xMax + 10)

    setPlotDim([xMin, xMax], [yMin, yMax])
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

    randomOrder = np.random.choice(range(len(inData)), len(inData))
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


def plotLinearDecisionBoundry(slope, bias, min, max):
    x = np.linspace(min, max, 100)
    y = slope * x + bias
    plt.plot(x, y, 'black')

def convergencePlot(epochs, mse, learningRate, color, seq):
    x = np.linspace(0, epochs, epochs)
    label = ""
    if seq:
        label = "Seq, Learning Rate: " + str(learningRate)
    else:
        label = "Batch, Learning Rate: " + str(learningRate)
    plt.plot(x, mse, color, label=label)

def showPlot(epochs, pointsPerClass):
    subtitle = "epochs: " + str(epochs) + ", points per class: " + str(
        pointsPerClass)
    plt.title("Delta Rule\n" + subtitle)
    plt.legend(loc="upper right")
    plt.show()

def plot3DMeshgridGaussianSamples(points):
    fig = plt.figure()
    ax = Axes3D(fig)
    for x, y, z, color in points:
        X, Y, Z = np.meshgrid(x, y, z)
        ax.plot_surface(X, Y, Z, color='red')

    xMin, xMax, yMin, yMax, zMin, zMax = extractExtremeValues(points)

    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    ax.set_zlim(zMin, zMax)

    plt.show()

def extractExtremeValues(points):
    xMin = np.min(np.concatenate(([p[0] for p in points]))) - 1
    xMax = np.max(np.concatenate(([p[0] for p in points]))) + 1

    yMin = np.min(np.concatenate(([p[1] for p in points]))) - 1
    yMax = np.max(np.concatenate(([p[1] for p in points]))) + 1

    zMin = np.min(np.concatenate(([p[2] for p in points]))) - 1
    zMax = np.max(np.concatenate(([p[2] for p in points]))) + 1

    return xMin, xMax, yMin, yMax, zMin, zMax