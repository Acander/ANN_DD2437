import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Removes the use of GPU

from tensorflow import Variable
from tensorflow import summary
from tensorflow import float64

def plotPoints(points, line=None, line2=None, label1="bsda", label2="asda"):
    '''
    Expects points to be a list of tuples:
        Tuple1: (xPoints, yPoints, color)
    '''
    plt.xlabel('t')
    plt.ylabel('y')

    for x, y, color, l in points:
        print(len(x), len(y), color, l)
        plt.plot(x, y, color, label=l, linewidth=2)

    xMin = np.min(np.concatenate(([p[0] for p in points])))
    xMax = np.max(np.concatenate(([p[0] for p in points])))

    yMin = np.min(np.concatenate(([p[1] for p in points])))
    yMax = np.max(np.concatenate(([p[1] for p in points])))

    '''
    if line is not None:
        s, b = line
        plotLinearDecisionBoundry(s, b, xMin - 10, xMax + 10, color="green", label=label1)

    if line2 is not None:
        s, b = line2
        plotLinearDecisionBoundry(s, b, xMin - 10, xMax + 10, color="purple", label=label2)
    '''

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
        ax.scatter(x, y, z, color)

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


def shuffleData(inData, labels):
    inData = inData.T
    labels = labels.T
    randomOrder = np.random.choice(range(len(inData)), len(inData), replace=False)
    inData = np.array([inData[i] for i in randomOrder])
    labels = np.array([labels[i] for i in randomOrder])
    return inData.T, labels.T


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


def plot3DMeshgridGaussianSamples(points):
    fig = plt.figure()
    ax = Axes3D(fig)
    for x, y, z, color in points:
        print(x.shape, y.shape, z.shape)
        ax.plot_trisurf(x, y, z, color='red', linewidth=0)
        # X, Y, Z = np.meshgrid(x, y, z)
        # ax.plot_surface(X, Y, Z, color='red')

    xMin, xMax, yMin, yMax, zMin, zMax = extractExtremeValues(points)
    #ax.set_xlim(xMin, xMax)
    #ax.set_ylim(yMin, yMax)
    #ax.set_zlim(zMin, zMax)

    plt.show()


def extractExtremeValues(points):
    xMin = np.min(np.concatenate(([p[0] for p in points]))) - 1
    xMax = np.max(np.concatenate(([p[0] for p in points]))) + 1

    yMin = np.min(np.concatenate(([p[1] for p in points]))) - 1
    yMax = np.max(np.concatenate(([p[1] for p in points]))) + 1

    zMin = np.min(np.concatenate(([p[2] for p in points]))) - 1
    zMax = np.max(np.concatenate(([p[2] for p in points]))) + 1

    return xMin, xMax, yMin, yMax, zMin, zMax


"""
    :param WeightData A tensor of weight values (aka a vector)
"""
def weightHistogramGenerator(weightData):
    summary.histogram("Weight data", weightData, step=None, buckets=None, description=None)

weightVectors = [0.1, 0.4, 0.3, 0.2, 0.2, 0.1, 0.2, 0.4, 0.3, 0.2]
weightHistogramGenerator(Variable(weightVectors, float64))