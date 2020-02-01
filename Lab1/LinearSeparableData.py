from Lab1.NeuralNetwork import FeedForwardNet, DenseLayer
from Lab1 import Utils, Activations, Losses
import numpy as np


def generateNetwork(loss=Losses.MSE(), lr=0.001):
    layers = [DenseLayer(2, 5, Activations.Sigmoid()), DenseLayer(5, 1, Activations.Sigmoid())]
    return FeedForwardNet(layers, loss, learningRate=lr)


def getDecisionBoundry(xWeight, yWeight, bias):
    k = - xWeight / yWeight
    m = -bias / yWeight
    return k, m


def trainNetwork(model, x, y, xValidation, yValidation):
    epochs = 0
    lastLoss = 9999
    losses = ([[], []], [[], []])
    while True:
        epochs += 1
        loss = model.fit(x, y, batchSize=1)
        lossVal = getMSEANN(model, xValidation, yValidation)
        losses[0][0].append(epochs)
        losses[0][1].append(loss)
        losses[1][0].append(epochs)
        losses[1][1].append(lossVal)
        if epochs % 200 == 0:
            print("Loss {}:".format(epochs), loss)

        # if lastLoss - loss < 0.00000001:
        if epochs == 2000:
            print("MSE:", loss)
            return epochs, losses

        lastLoss = loss


def getMSEANN(model, x, y):
    predictions = model.forwardPass(x)
    MSE = np.mean((predictions - y) ** 2)
    return MSE


def getAccuracyANN(model, x, y):
    predictions = model.predict(x)
    numCorrect = np.sum(np.equal(predictions, y).astype(int))
    freqCorr = numCorrect / np.size(y)
    return freqCorr


def splitData(p1, p2, p3, p4, N, eightyTwenty):
    nCluster = int(N / 4)
    N = int(N * 0.75)

    indices = np.arange(len(p1[0]))
    np.random.shuffle(indices)
    if not eightyTwenty:
        training = indices[:int(nCluster * 2 / 2)]
        validation = indices[int(-nCluster * 2 / 2):]

        p1Validation = (p1[0][validation], p1[1][validation], p1[2])
        p1 = (p1[0][training], p1[1][training], p1[2])
        # p2 = (p2[0][training], p2[1][training], p2[2])

        p1 = (np.concatenate((p1[0], p2[0])), np.concatenate((p1[1], p2[1])), p1[2])
        p2 = (np.concatenate((p3[0], p4[0])), np.concatenate((p3[1], p4[1])), p4[2])

        labelsValidation = np.ones((1, len(p1Validation[0])))
    else:
        training1 = indices[:int(nCluster * 0.8)]
        training2 = indices[:int(nCluster * 0.2)]
        val1 = indices[-int(nCluster * 0.2):]
        val2 = indices[-int(nCluster * 0.8):]

        # p1Validation = (p1[0][val1], p1[1][val1], p1[2])

        p1Validation = (np.concatenate((p1[0][val1], p2[0][val2])),
                        np.concatenate((p1[1][val1], p2[1][val2])),
                        p1[2])

        labelsValidation = np.ones((1, len(p1Validation[0])))

        p1 = (np.concatenate((p1[0][training1], p2[0][training2])),
              np.concatenate((p1[1][training1], p2[1][training2])),
              p1[2])
        p2 = (np.concatenate((p3[0], p4[0])), np.concatenate((p3[1], p4[1])), p4[2])
        # print("Num training points:", len(p1[0]) + len(p2[0]))
        # print("Num validation points:", len(p1Validation[0]))
        # Utils.plotPoints([p1, p2])
        # Utils.plotPoints([p1Validation])

        # xValidation = []

    xValidation = np.vstack([p1Validation[0], p1Validation[1]])

    return p1, p2, xValidation, labelsValidation, N


def performTest(p1, p2, p3, p4, N_CLASS, N, color='red', eightyTwenty=False):
    # Utils.plotPoints([p1, p2, p3, p4])
    learningRate = 0.004
    model = generateNetwork(lr=learningRate)

    p1, p2, xValidation, labelsValidation, N = splitData(p1, p2, p3, p4, N, eightyTwenty)

    # Formatting to match input
    x1 = np.vstack([p1[0], p1[1]])
    x2 = np.vstack([p2[0], p2[1]])

    # Generate Labels
    labels1 = np.ones((1, len(p1[0])))
    labels2 = np.zeros((1, len(p2[0])))

    x, y = Utils.stackAndShuffleData([x1, x2], [labels1, labels2])

    # Test backprop
    epochsBP, losses = trainNetwork(model, x, y, xValidation, labelsValidation)

    print("AccuracyTraining:", getAccuracyANN(model, x, y))
    print("AccuracyValidation:", getAccuracyANN(model, xValidation, labelsValidation))
    print("MSETraining:", getMSEANN(model, x, y))
    print("MSEValidation:", getMSEANN(model, xValidation, labelsValidation))

    Utils.plotDecisionBoundary(model, [p1, p2])

    return losses


if __name__ == '__main__':
    import Lab1.PerceptronLearning

    N_CLASS = 50
    N = 200
    # p1, p2 = Utils.generateData(N_CLASS, [[4, 4], [5, 5]], ['ro', 'bo'], [1, 1])

    p1, p2, p3, p4 = Utils.generateData(N_CLASS, [[-1, -0.3], [1, 0.3], [0, -0.1], [0, -0.1]], ['ro', 'ro', 'bo', 'bo'],
                                        [0.2, 0.2, 0.3, 0.3])
    '''

    # p1 = (np.random.choice(p1[0], int(N * 0.8 * 0.25), replace=False),
    #       np.random.choice(p1[1], int(N * 0.8 * 0.25), replace=False), p1[2])

    # p2 = (np.random.choice(p2[0], int(N * 0.2 * 0.25), replace=False),
    #       np.random.choice(p2[1], int(N * 0.2 * 0.25), replace=False), p2[2])

    p1 = (np.concatenate((p1[0], p2[0])), np.concatenate((p1[1], p2[1])), p1[2])
    p2 = (np.concatenate((p3[0], p4[0])), np.concatenate((p3[1], p4[1])), p4[2])

    # p1 = (np.random.choice(p1[0], int(N * 0.5 * 0.75), replace=False),
    #       np.random.choice(p1[1], int(N * 0.5 * 0.75), replace=False), p1[2])

    # p2 = (np.random.choice(p2[0], int(N * 0.5 * 0.75), replace=False),
    #       np.random.choice(p2[1], int(N * 0.5 * 0.75), replace=False), p2[2])

    N = int(N * 0.75)

    indices = np.arange(len(p1[0]))
    np.random.shuffle(indices)
    training = indices[:int(N_CLASS * 2 / 2)]  # N_CLASS is actually N_CLUSTER =)
    validation = indices[int(-N_CLASS * 2 / 2):]

    p1Validation = (p1[0][validation], p1[1][validation], p1[2])
    p1 = (p1[0][training], p1[1][training], p1[2])
    # p2 = (p2[0][training], p2[1][training], p2[2])

    learningRate = 0.004
    model = generateNetwork(lr=learningRate)

    # Formatting to match input
    x1 = np.vstack([p1[0], p1[1]])
    x2 = np.vstack([p2[0], p2[1]])

    xValidation = np.vstack([p1Validation[0], p1Validation[1]])
    labelsValidation = np.ones((1, len(xValidation[0])))

    print("Training data Shape:", x1.shape, x2.shape)
    # Generate Labels
    labels1 = np.ones((1, len(p1[0])))
    labels2 = np.zeros((1, len(p2[0])))
    print("Traning labels Shape:", labels1.shape, labels2.shape)

    x, y = Utils.stackAndShuffleData([x1, x2], [labels1, labels2])
    print("Final data shape:", x.shape, y.shape)

    # Test backprop
    epochsBP = trainNetwork(model, x, y)

    # Test perceptron learning
    xBiasAdded = np.vstack([x, np.ones(N)])
    yTransformed = y * 2 - 1
    perceptron, epochsPL = Lab1.PerceptronLearning.getTrainedModel(p1, p2, xBiasAdded, yTransformed, learningRate,
                                                                   plotProgress=False, printProgress=False)
    w1, w2, b = model.layers[0].weights[0]
    kBP, mBP = getDecisionBoundry(w1, w2, b)
    # w1, w2, b = perceptron.weights
    # kPL, mPL = getDecisionBoundry(w1, w2, b)
    # Utils.plotPoints([p1, p2], [kBP, mBP], [kPL, mPL],
    #                  label1="BackPropagation, t=" + str(epochsBP), label2="PerceptronLearning, t=" + str(epochsPL))

    # Utils.plotPoints([p1, p2], [kBP, mBP], label1="BackPropagation, t=" + str(epochsBP))
    print("AccuracyTraining:", getAccuracyANN(model, x, y))
    print("AccuracyValidation:", getAccuracyANN(model, xValidation, labelsValidation))
    Utils.plotDecisionBoundary(model, [p1, p2])
    '''
    lossesList = []
    colors1 = ['red', 'blue', 'green', 'purple', 'brown', 'yellow']
    colors2 = ['darkred', 'darkblue', 'darkgreen', 'purple', 'brown', 'yellow']
    # labels = ['50% pruned', '80%/20% pruned']
    for i in range(2):
        (loss, lossValidation) = performTest(p1, p2, p3, p4, N_CLASS, N, eightyTwenty=bool(i))
        lossesList.append((loss[0], loss[1], colors1[i]))
        lossesList.append((lossValidation[0], lossValidation[1], colors2[i]))

    Utils.plotPoints(lossesList)
