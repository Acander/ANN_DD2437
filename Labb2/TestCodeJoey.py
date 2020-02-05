import numpy as np

from Labb2 import CL
from Labb2.DataHandler import generateData, plotPoints
from Labb2.RBFFunc import RBF
from Labb2.RBFNet import RadialBasisFunctionNetwork

if __name__ == '__main__':
    xTrain, yTrain, xTest, yTest = generateData(False, noiseVariance=0.0)
    numHidden = 4
    model = RadialBasisFunctionNetwork(1, 1, numHidden, np.random.uniform(0, 6 * np.pi, (numHidden, 1)), RBF,
                                       l1Dist=False)
    prevCentroids, centroids = CL.learnClusters(xTrain, model.centroids, multiWinner=False)
    prevCentroids, centroidsMW = CL.learnClusters(xTrain, prevCentroids, multiWinner=True)
    colors = ["bo", "ro", "yo", "go"]
    labels = ["X", "Pre-CL", "Post-CL", "Post-CL-MultiWinner"]
    sizes = [5, 10, 10, 10]
    plotPoints([np.reshape(xTrain, (len(xTrain))), prevCentroids, centroids, centroidsMW], colors, labels, sizes)
