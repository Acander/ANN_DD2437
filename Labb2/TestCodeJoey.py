import numpy as np

from Labb2 import CL
from Labb2.DataHandler import generateData, plotPoints, evaluateModel
from Labb2.RBFFunc import RBF
from Labb2.RBFNet import RadialBasisFunctionNetwork


# lr=0.01
# sigma=0.5
# numHidden=500
# placed RBF with np.arange
# 18880 MAE 0.013484278389808977

if __name__ == '__main__':
    xTrain, yTrain, xTest, yTest = generateData(False, shuffle=False, noiseVariance=0.0)
    numHidden = 500
    model = RadialBasisFunctionNetwork(1, 1, numHidden, np.random.uniform(0, 2 * np.pi, (numHidden, 1)), RBF,
                                       lr=0.1, l1Dist=False)

    # prevCentroids, centroids = CL.learnClusters(xTrain, model.centroids, multiWinner=True)

    model.centroids = np.reshape(np.arange(0, 2 * np.pi, 2 * np.pi / numHidden), (numHidden, 1))
    colors = ["bo", "ro", "yo", "go"]
    labels = ["X", "Pre-CL", "Post-CL", "Post-CL-MultiWinner"]
    sizes = np.array([5, 10, 10, 10]) / 2.0
    # plotPoints([np.reshape(xTrain, (len(xTrain))), centroids], colors, labels, sizes)

    for i in range(10000000):
        if i % 1000 == 0:
            model.lr /= 2
        loss = model.fit(xTrain, yTrain)

        if i % 10 == 0:
            # print("MSE", loss)
            print("ResidualError:", i)
            print("Train:", evaluateModel(model, xTrain, yTrain, True))
            print("Test:", evaluateModel(model, xTest, yTest, True))
    '''
    prevCentroids, centroids = CL.learnClusters(xTrain, model.centroids, multiWinner=False)
    prevCentroids, centroidsMW = CL.learnClusters(xTrain, prevCentroids, multiWinner=True)
    colors = ["bo", "ro", "yo", "go"]
    labels = ["X", "Pre-CL", "Post-CL", "Post-CL-MultiWinner"]
    sizes = [5, 10, 10, 10]
    plotPoints([np.reshape(xTrain, (len(xTrain))), prevCentroids, centroids, centroidsMW], colors, labels, sizes)
    '''
