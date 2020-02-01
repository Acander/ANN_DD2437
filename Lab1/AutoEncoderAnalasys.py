from Lab1 import AutoEncoder, Utils
import numpy as np
import pickle


def _dist(x, y):
    return np.sqrt((x - y) ** 2)


def getMeanDistance(weights):
    distances = []
    for i, w1 in enumerate(weights):
        for j, w2 in enumerate(weights):
            if (i != j):
                distances.append(_dist(w1, w2))
    print(len(distances))
    return np.mean(distances)


with open('ModelWeights2.pkl', 'rb') as fp:
    weights = pickle.load(fp)
with open('ModelRandomWeights2.pkl', 'rb') as fp:
    randomWeights = pickle.load(fp)

'''
print(weights[0].T[:-1].shape)
weights = [w for w in weights[0].T[:-1]]
print("Mean:", getMeanDistance(weights))
'''
model = AutoEncoder.generateNetwork()
for l, w in zip(model.layers, weights):
    l.weights = w

print(model.layers[0].weights.T.shape)
points = []
for p in model.layers[0].weights.T:
    points.append([[c] for c in p] + ['ro'])
print(points)
Utils.plotPoints(points)

'''
'''
