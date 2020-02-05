import numpy as np


# Returns a sorted list with indices if returnAll, otherwise just the index of the closest centroid
def closestCentroid(sample, centroids, returnAll=False):
    # print("centroid:", np.shape(centroids))
    # print("sample:", np.shape(sample))
    absDistances = np.linalg.norm(centroids - sample, axis=1)
    return np.argsort(absDistances) if returnAll else np.argmin(absDistances)


def learnClusters(X, centroids, iterations=10000, learningRate=0.02, multiWinner=True):
    invAscSeq = np.square(1 / np.reshape(np.arange(1, len(centroids) + 1), (len(centroids), 1)))
    # print(invAscSeq)
    X = np.reshape(X, (len(X)))
    # print(centroids)
    prevCentroids = np.copy(centroids)
    for i in range(iterations):
        trainingSample = np.random.choice(X)
        closestIdx = closestCentroid(trainingSample, centroids, returnAll=multiWinner)
        centroids[closestIdx] += learningRate * (trainingSample - centroids[closestIdx]) * (
            invAscSeq if multiWinner else 1)
        # 1 / (np.arange(len(closestIdx)) + 1))
        # print(trainingSample)
    # print(centroids)
    return prevCentroids, centroids
