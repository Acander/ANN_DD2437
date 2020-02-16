import numpy as np
import matplotlib.pyplot as plt

from Labb3 import Utils, DataHandler
from Labb3.HopfieldNetwork import HopsNet

if __name__ == '__main__':

    '''
    patterns = [
        [-1, -1, 1, -1, 1, -1, -1, 1],
        [-1, -1, -1, -1, -1, 1, -1, -1],
        [-1, 1, 1, -1, -1, 1, -1, 1],
    ]
    noisyPatterns = [
        [1, -1, 1, -1, 1, -1, -1, 1],
        [1, 1, -1, -1, -1, 1, -1, -1],
        [1, 1, 1, -1, 1, 1, -1, 1],
    ]
    '''
    allPatterns = DataHandler.importAllPicData()
    patterns = allPatterns[0:3]
    P = len(patterns[0])

    model = HopsNet(P)
    model.setWeights(patterns, setDiagZero=False)
    model.weights = Utils.generateRandomWeightMatrix(P, True)

    prediction, epochs, history = model.sequentialPredict(allPatterns[0], numIteration=100000)
    print(epochs)
    # energies = [Utils.energy(model.weights, history[i]) for i in range(0, len(history), 1000)]
    # print(energies)
    energyInit = Utils.energy(model.weights, history[0])  # * 10 ** (-6)
    energyLast = Utils.energy(model.weights, prediction)  # * 10 ** (-6)
    print(energyInit)
    print(energyLast)
    # print(model.weights)

    '''
    # print("Patterns:")
    # for i, x in enumerate(patterns):
    # prediction, epochs, history = model.predict(allPatterns[10])
    prediction, _, history = model.sequentialPredict(allPatterns[9], numIteration=7000)
    print("Cacling energies...")
    stepSize = 50
    energies = [Utils.energy(model.weights, history[i]) * 10 ** (-6) for i in range(0, len(history), stepSize)]
    print("Energies calculated")
    plt.plot(range(0, len(history), stepSize), energies, label="p10")
    prediction, _, history = model.sequentialPredict(allPatterns[10], numIteration=7000)
    energies = [Utils.energy(model.weights, history[i]) * 10 ** (-6) for i in range(0, len(history), stepSize)]
    plt.plot(range(0, len(history), stepSize), energies, label="p11")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Energy * 10^-6")
    plt.show()
    '''
    # print("epochs:", epochs)
    # print("energy:", Utils.energy(model.weights, allPatterns[0]))
    '''
    p1: -1473936 * 10^-6 => -1,4
    p2: -1398416
    p3: -1497344
    p11, incorrect attraction found by sync: -1634316
    p10: -425964
    p11: -177664
    '''
    # print(i, np.sum(np.abs(prediction - x)) / 2)
    # for dp in [history[i] for i in range(0, len(history), 1000)]:
        # DataHandler.plotDatapoint(dp)

    '''
    predict10, _ = model.predict(allPatterns[9])
    print("prediction of 10:", np.sum(np.abs(predict10 - allPatterns[0])) / 2)

    predict11, _ = model.predict(allPatterns[10])
    print("prediction of 11:", np.sum(np.abs(predict11 - allPatterns[1])) / 2)

    predict11, _ = model.predict(allPatterns[10])
    print("prediction of 11:", np.sum(np.abs(predict11 - allPatterns[2])) / 2)
    '''
    '''

    print("Noisy:")
    for i, n in enumerate(noisyPatterns):
        final, epochs = model.predict(np.array(n))
        print(np.sum(np.abs(final - patterns[i])), epochs)
    # print("\n", final)
    '''

    '''
    possibleSeqs = Utils.allPossibleSequences(8)
    print(possibleSeqs)
    attractions = []
    for i, x in enumerate(possibleSeqs):
        prediction, epochs = model.predict(np.array(x))

        if not Utils.existsInList(attractions, prediction):
            attractions.append(prediction)

    attractions = np.array(attractions)
    print(attractions)
    print(len(attractions))
    for a in attractions:
        prediction, epochs = model.predict(a)
        if not np.array_equal(a, prediction):
            print("DAMN")
    '''
