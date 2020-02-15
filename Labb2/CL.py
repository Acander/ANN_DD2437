import numpy as np

# Returns a sorted list with indices if returnAll, otherwise just the index of the closest centroid
from Labb2.DataHandler import generateData, plotPointsXY, evaluateModel
from Labb2.DataSetHandler import importTrainingBallisticData, importTestBallisticData
from Labb2.TFNet import RadialBasisFunctionNetwork


def closestCentroid(sample, centroids, returnAll=False):
    # print("centroid:", np.shape(centroids))
    # print("sample:", np.shape(sample))
    absDistances = np.linalg.norm(centroids - sample, axis=1)
    return np.argsort(absDistances) if returnAll else np.argmin(absDistances)


def learnClusters(X, centroids, iterations=10000, learningRate=0.02, multiWinner=True):
    invAscSeq = np.square(1 / np.reshape(np.arange(1, len(centroids) + 1), (len(centroids), 1)))
    # print(invAscSeq)
    # print(X)
    # X = np.reshape(X, (len(X)))
    # print(centroids)
    prevCentroids = np.copy(centroids)
    for i in range(iterations):
        # trainingSample = np.random.choice(X)
        trainingSample = X[np.random.randint(0, len(X))]
        closestIdx = closestCentroid(trainingSample, centroids, returnAll=multiWinner)
        centroids[closestIdx] += learningRate * (trainingSample - centroids[closestIdx]) * (
            invAscSeq if multiWinner else 1)
        # 1 / (np.arange(len(closestIdx)) + 1))
        # print(trainingSample)
    # print(centroids)
    return prevCentroids, centroids


def extractBallisticData(data):
    X = []
    Y = []
    for i in range(len(data)):
        x, y = data[i]
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


'''
With CL:
ResidualError: 50000
Train: 0.02122817
Test: 0.024123956


Without CL and uniformly placed in a larger area:
ResidualError: 50000
Train: 0.051645212
Test: 0.056954958
'''

if __name__ == '__main__':
    # yTrain, xTest, yTest = generateData(False, shuffle=False, noiseVariance=0.0)
    numHidden = 50
    # centroids = np.reshape(np.arange(0, 2 * np.pi, 2 * np.pi / numHidden), (numHidden, 1))
    centroids = np.random.uniform(0, 10, (numHidden, 2))
    # newX = []
    # for i in range(len(xTrain)):
    # newX.append([xTrain[i][0], xTrain[i][0]])
    # newX = np.array(newX)
    # print(newX)
    # print(centroids)
    X, Y = extractBallisticData(importTrainingBallisticData())
    # print(importTestBallisticData())
    XTest, YTest = extractBallisticData(importTestBallisticData())
    # print(XTest)
    # print(YTest)
    X = X.astype('float32')
    Y = Y.astype('float32')
    XTest = XTest.astype('float32')
    YTest = YTest.astype('float32')
    # print(X)
    # learnClusters(X, centroids, multiWinner=True)
    plotPointsXY([[X[:, 0], X[:, 1]], [centroids[:, 0], centroids[:, 1]]], ["x", "centroids"], drawPoints=True,
                 drawLines=False)
    SIGMA = 1
    model = RadialBasisFunctionNetwork(2, 2, numHidden, centroids, SIGMA)
    for i in range(1, 100000):
        # if i % 1000 == 0:
        #     model.lr *= 0.98
        loss = model.fit(X, Y)

        if i % 100 == 0:
            # print("MSE", loss)
            print("ResidualError:", i)
            print("Train:", evaluateModel(model, X, Y, True))
            print("Test:", evaluateModel(model, XTest, YTest, True))
