from Lab1 import Utils
from Labb3 import DataHandler, HopfieldNetwork, NoiseTests, Utils as Utils3
import numpy as np
import pickle


def capacityLimit(model, patterns, theta=0):
    # for i in range(1, len(patterns) + 1, 5):
    model.setWeights(patterns, setDiagZero=True, removeBias=True)
    score = [0, 0]
    for p in patterns:
        final, epochCounter, history = model.predict(p, theta)
        diffScore = NoiseTests._getDifferenceInpatterns(final, p)
        score[0 if diffScore == 0 else 1] += 1

    ratio = score[0] / (score[0] + score[1])

    return ratio


def _testModelOnPatterns(model, patterns, theta=0):
    score = [0, 0]
    for p in patterns:
        # final, epochCounter, history = model.predict(Utils3.flipBits(p, 3))
        final, epochCounter, history = model.sequentialPredict(p, theta)
        diffScore = NoiseTests._getDifferenceInpatterns(final, p)
        score[0 if diffScore == 0 else 1] += 1

    ratio = score[0] / (score[0] + score[1])
    print(score, ratio)
    return ratio


def generateRandomPatterns(size=1024, numPatterns=10):
    patterns = []
    for i in range(numPatterns):
        patterns.append(np.sign(np.random.normal(0, 1, size) + 0.5))
        # patterns.append(np.array([1 if np.random.random() > 0.5 else -1 for j in range(size)]))
    return patterns


def plotData():
    fileNames = {
        'Normal': 'BiasCapacityNoramDiag.pkl',
        'NormalZero': 'BiasCapacityZeroDiag.pkl',
        'Noise': 'BiasCapacityNoramDiagNoise.pkl',
        'NoiseZero': 'BiasCapacityZeroDiagNoise.pkl',
    }

    results = []
    colors = ['green', 'red', 'orange', 'blue']
    for i, f in enumerate(fileNames):
        print(f)
        with open(fileNames[f], 'rb') as fp:
            data = pickle.load(fp)
        results.append((range(len(data)), data, colors[i], f))

    Utils.plotPoints(results)


if __name__ == '__main__':
    plotData()
    # patterns = DataHandler.importAllPicData()
    '''
    patterns = generateRandomPatterns(size=100, numPatterns=300)

    for i, p in enumerate(patterns):
        dists = []
        for j in range(len(patterns)):
            if (i != j):
                dists.append(NoiseTests._getDifferenceInpatterns(p, patterns[j]))
        print(i, np.mean(dists), np.min(dists))

    results = []
    for i in range(1, len(patterns) + 1):
        print("*****", i)
        model = HopfieldNetwork.HopsNet(100)
        model.setWeights(patterns[:i], setDiagZero=True)
        # print(np.diag(model.weights))
        # results.append((np.mean(np.abs(model.weights)), np.std(model.weights)))
        results.append(_testModelOnPatterns(model, patterns[:i]))

    with open('BiasCapacityZeroDiag.pkl', 'wb') as fp:
        pickle.dump(results, fp)
    '''

    # r1 = [r[0] for r in results]
    # r2 = [r[1] for r in results]
    # Utils.plotPoints([(range(600), r1, 'green', 'mean'), (range(600), r2, 'red', 'std')])
    # Utils.plotPoints([(range(600), results, 'green', 'mean')])
