from Lab1.NeuralNetwork import FeedForwardNet, DenseLayer
from Lab1 import Utils, Activations, Losses
import numpy as np


def generateNetwork(loss=Losses.MSE(), lr=0.001):
    layers = [DenseLayer(2, 1, Activations.Sigmoid())]
    return FeedForwardNet(layers, loss, learningRate=lr)


def getDecisionBoundry(xWeight, yWeight, bias):
    ''' Trash formula
    xMax = bias / xWeight
    yMax = bias / yWeight
    slope = -(yMax / xMax)
    return slope, bias
    '''
    k = - xWeight / yWeight
    m = -bias / yWeight
    return k, m


def trainNetwork(model, x, y):
    epochs = 0
    lastLoss = 9999
    while True:
        epochs += 1
        loss = model.fit(x, y, batchSize=-1)
        # print("Loss {}:".format(epochs), loss)

        if lastLoss - loss < 0.00001:
            return epochs

        lastLoss = loss


if __name__ == '__main__':
    import Lab1.PerceptronLearning

    N_CLASS = 50
    p1, p2 = Utils.generateData(N_CLASS, [[3, 3], [4, 4]], ['ro', 'bo'], [1, 1])
    # Utils.plotPoints([p1, p2])
    learningRate = 0.001
    model = generateNetwork(lr=learningRate)

    # Formatting to match input
    x1 = np.vstack([p1[0], p1[1]])
    x2 = np.vstack([p2[0], p2[1]])
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
    xBiasAdded = np.vstack([x, np.ones(N_CLASS * 2)])
    yTransformed = y * 2 - 1
    # perceptron, epochsPL = Lab1.PerceptronLearning.getTrainedModel(p1, p2, xBiasAdded, yTransformed, learningRate,
    #                                                                plotProgress=False, printProgress=False)
    w1, w2, b = model.layers[0].weights[0]
    kBP, mBP = getDecisionBoundry(w1, w2, b)
    # w1, w2, b = perceptron.weights
    # kPL, mPL = getDecisionBoundry(w1, w2, b)
    # Utils.plotPoints([p1, p2], [kBP, mBP], [kPL, mPL],
    #                 label1="BackPropagation, t=" + str(epochsBP), label2="PerceptronLearning, t=" + str(epochsPL))

    Utils.plotPoints([p1, p2], [kBP, mBP],
                     label1="BackPropagation, t=" + str(epochsBP))
