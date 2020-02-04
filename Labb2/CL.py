import numpy as np


def closestCentroid(sample, centroids):
    closestIdx = np.argmin(np.abs(centroids - sample))
    # print(np.abs(centroids - sample), "Sample:", sample)
    # print("Clusters:", centroids)
    # print("Sample:", sample, ", Closest:", closest)
    return closestIdx


def learnClusters(X, centroids, iterations=100000, learningRate=0.5):
    X = np.reshape(X, (len(X)))
    print(centroids)
    for i in range(iterations):
        trainingSample = np.random.choice(X)
        closestIdx = closestCentroid(trainingSample, centroids)
        centroids[closestIdx] += learningRate * (trainingSample - centroids[closestIdx])
        # print(trainingSample)
    print(centroids)
