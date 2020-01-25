from NeuralNetwork import FeedForwardNet, DenseLayer
import Utils, Activations, Losses
import numpy as np

EPOCHS = 50
POINTS_PER_CLASS = 100


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


if __name__ == '__main__':
    """np.random.seed(50)
    convergenceColors = ['red', 'green', 'yellow', 'blue', 'pink', 'purple']
    learningRates = [0.001, 0.002, 0.004, 0.001, 0.002, 0.004]
    mse = np.zeros(EPOCHS)
    seq = True
    batchSize = 1

    p1, p2 = Utils.generateData(POINTS_PER_CLASS, [[0, 0], [4, 4]], ['ro', 'bo'], [0.5, 0.5])
    model = generateNetwork()
    weights = [l.weights.copy() for l in model.layers]
    for i in range(6):
        for w, l in zip(weights, model.layers):
            l.weights = w.copy()
        model.lr = learningRates[i]
        # Utils.plotPoints([p1, p2])

        # Formatting to match input
        x1 = np.vstack([p1[0], p1[1]])
        x2 = np.vstack([p2[0], p2[1]])
        # print("Traning data Shape:", x1.shape, x2.shape)
        # Generate Labels
        labels1 = np.ones((1, len(p1[0])))
        labels2 = np.zeros((1, len(p2[0])))
        # print("Traning labels Shape:", labels1.shape, labels2.shape)

        mse = np.zeros(EPOCHS)
        x, y = Utils.stackAndShuffleData([x1, x2], [labels1, labels2])
        # print("Final data shape:", x.shape, y.shape)

        for j in range(EPOCHS):
            mse[j] = model.fit(x, y, batchSize=batchSize)
            # print("Loss {}:".format(i), mse[i])

        w1, w2, b = model.layers[0].weights[0]
        line = getDecisionBoundry(w1, w2, b)
        if i > 2:
            seq = False
            batchSize = -1

        Utils.convergencePlot(EPOCHS, mse, learningRates[i], convergenceColors[i], seq)

    # Utils.plotPoints([p1, p2], EPOCHS, BATCH_SIZE, POINTS_PER_CLASS, line)

    Utils.showPlot(EPOCHS, POINTS_PER_CLASS)"""
    Utils.plot3DMeshgridGaussianSamples()
