import numpy as np

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

    print(model.weights)

    print("Patterns:")
    # for i, x in enumerate(patterns):
    prediction, epochs, history = model.predict(allPatterns[10])
    # prediction, epochs, history = model.sequentialPredict(allPatterns[2], numIteration=10000)
    print("epochs:", epochs)
    # print(i, np.sum(np.abs(prediction - x)) / 2)
    for dp in [history[i] for i in range(0, len(history), 1)]:
        DataHandler.plotDatapoint(dp)

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