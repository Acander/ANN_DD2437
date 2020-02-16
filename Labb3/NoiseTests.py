from Lab1 import Utils
from Labb3 import DataHandler, HopfieldNetwork
import numpy as np


def _getDifferenceInpatterns(a, b):
    return np.sum(np.abs(a - b)) / 2


def generateNoiseToPattern(pattern, noiseRatio=0.5):
    noisePattern = pattern.copy()
    noiseIndicies = np.random.choice(range(len(pattern)), int(len(pattern) * noiseRatio), replace=False)
    for i in noiseIndicies:
        noisePattern[i] = 1 if np.random.random() > 0.5 else -1
    return noisePattern


if __name__ == '__main__':
    patterns = DataHandler.importAllPicData()
    model = HopfieldNetwork.HopsNet(1024)
    model.setWeights(patterns[:3])
    print(model.weights)

    allResults = []
    for epoch in range(500):
        print(epoch)
        results = {}
        for j in range(3):
            ratio = 0.8

            nPattern = generateNoiseToPattern(patterns[j], ratio)
            x, epoch, history = model.predict(nPattern)
            results[j] = [_getDifferenceInpatterns(h, patterns[j]) for h in history]
            # results[j]['ReconstructionScore'].append(_getDifferenceInpatterns(x, patterns[j]))

        allResults.append(results)

    r = {}
    for j in range(3):
        sizes = [len(rez[j]) for rez in allResults]
        maxLen = np.max(sizes)
        print(j, np.max(sizes), np.std(sizes), np.mean(sizes))
        temp = []
        for rez in allResults:
            for i in range(maxLen - len(rez[j])):
                rez[j].append(rez[j][-1])
            temp.append(rez[j])

        temp = np.array(temp)
        print(j, np.std(temp, axis=0), np.mean(temp, axis=0))
        r[j] = np.mean(np.array([rez[j] for rez in allResults]), axis=0)
        print(r[j].shape)

    print(r)
    labels = {0: 'P1', 1: 'P2', 2: 'P3'}
    colors = {0: 'red', 1: 'green', 2: 'blue'}
    points = [(range(len(r[k])), r[k], colors[k], labels[k]) for k in range(3)]
    Utils.plotPoints(points)

'''
allResults = []
for epoch in range(200):
    print("Epoch:", epoch)
    results = {}
    for j in range(3):
        results[j] = {'Ratios': [], 'ReconstructionScore': []}

        for i in range(101):
            ratio = i / 100
            results[j]['Ratios'].append(ratio)

            nPattern = generateNoiseToPattern(patterns[j], ratio)
            x, epoch, history = model.predict(nPattern)
            results[j]['ReconstructionScore'].append(_getDifferenceInpatterns(x, patterns[j]))
    allResults.append(results)

    r = {}
    for j in range(3):
        meanRatios = np.mean(np.array([r[j]['Ratios'] for r in allResults]), axis=0)
        meanScores = np.mean(np.array([r[j]['ReconstructionScore'] for r in allResults]), axis=0)
        r[j] = {'Ratios': meanRatios, 'ReconstructionScore': meanScores}
    
    labels = {0: 'P1', 1: 'P2', 2: 'P3'}
    colors = {0: 'red', 1: 'green', 2: 'blue'}
    points = [(r[k]['Ratios'], r[k]['ReconstructionScore'], colors[k], labels[k]) for k in range(3)]
    Utils.plotPoints(points)
'''
