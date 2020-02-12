import numpy as np

from Labb2 import CL
from Labb2.DataHandler import generateData, plotPoints, evaluateModel, plotPointsXY
from Labb2.RBFFunc import RBF
from Labb2.RBFNet import RadialBasisFunctionNetwork


# lr=0.01
# sigma=0.5
# numHidden=500
# placed RBF with np.arange
# 18880 MAE 0.013484278389808977

'''
ResidualError: 32460
Train: 0.009840412721151343
Test: 0.010249602660420477


sigma 0.3:
ResidualError: 20040
Train: 0.008290703034550862
Test: 0.008462745568045535

sigma 0.1:
numHidden=60
ResidualError: 34650
Train: 0.008803671466040287
Test: 0.009648410206915441

'''

if __name__ == '__main__':
    xTrain, yTrain, xTest, yTest = generateData(False, shuffle=False, noiseVariance=0.0)
    # plotPointsXY(xTrain, yTrain)
    numHidden = 4
    model = RadialBasisFunctionNetwork(1, 1, numHidden, np.random.uniform(0, 2 * np.pi, (numHidden, 1)), RBF,
                                       lr=0.7, l1Dist=False)

    # prevCentroids, centroids = CL.learnClusters(xTrain, model.centroids, multiWinner=True)

    # model.centroids = np.reshape(np.arange(0, 2 * np.pi, 2 * np.pi / numHidden), (numHidden, 1))
    model.centroids = np.random.uniform(0, 10, (numHidden, 1))
    colors = ["bo", "ro", "yo", "go"]
    labels = ["X", "Pre-CL", "Post-CL", "Post-CL-MultiWinner"]
    sizes = np.array([5, 10, 10, 10]) / 2.0
    # plotPoints([np.reshape(xTrain, (len(xTrain))), centroids], colors, labels, sizes)

    '''
    for i in range(1, 100000):
        # if i % 1000 == 0:
        #     model.lr *= 0.98
        loss = model.fit(xTrain, yTrain)

        if i % 100 == 0:
            # print("MSE", loss)
            print("ResidualError:", i)
            print("Train:", evaluateModel(model, xTrain, yTrain, True))
            print("Test:", evaluateModel(model, xTest, yTest, True))

    yPred = model.predict(xTrain)
    '''
    # plotPointsXY([[xTrain, yTrain], [xTrain, yPred]], ["True", "Approx"])

    np.random.seed(1337)
    prevCentroids, centroids = CL.learnClusters(xTrain, model.centroids, multiWinner=False)
    prevCentroids, centroidsMW = CL.learnClusters(xTrain, prevCentroids, multiWinner=True)
    colors = ["bo", "ro", "yo", "go"]
    # labels = ["X", "Pre-CL", "Post-CL", "Post-CL-MultiWinner"]
    labels = ["X", "Pre-CL", "Post-CL"]
    sizes = [3, 8, 8, 8]
    # plotPoints([np.reshape(xTrain, (len(xTrain))), prevCentroids, centroids, centroidsMW], colors, labels, sizes)
    plotPoints([np.reshape(xTrain, (len(xTrain))), prevCentroids, centroids], colors, labels, sizes)

