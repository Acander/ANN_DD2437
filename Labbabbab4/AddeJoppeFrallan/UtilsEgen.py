import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

hToPlotIndices = None


def meanReconstLossOnParts(model, dataParts):
    sizes = [len(d) for d in dataParts]
    losses = [meanReconstLossTestSet(model, p).numpy() for p in dataParts]
    return np.sum(losses * (sizes / np.sum(sizes)))  # Weighted average, by the length in each part


def meanReconstLossTestSet(rbm, testSet):
    ph0, h0 = rbm.get_h_given_v(testSet)
    pv1, v1 = rbm.get_v_given_h(h0)
    return meanReconstLoss(testSet, v1)


def meanReconstLoss(v0, v1):
    return tf.reduce_mean(tf.reduce_sum(tf.abs(v1 - v0), axis=1)) / v1.shape[1]


def plotLearningCurves(lossesList, labels):
    X = np.arange(len(lossesList))
    for i in range(lossesList):
        plt.plot(X, lossesList[i], labels[i])

    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.show()


def plotWeights(weights, epochId):
    image_size = (28, 28)
    hLen = weights.shape[1]
    numToPlot = 10

    global hToPlotIndices
    if hToPlotIndices is None:
        hToPlotIndices = np.random.choice(np.arange(hLen), numToPlot, replace=False).tolist()
        # hToPlotIndices = [int(h) for h in hToPlotIndices]

    # if epoch % self.rf["period"] == 0 and self.is_bottom:
    from Labbabbab4.codeAlaPawel.util import viz_rf
    print(hToPlotIndices)
    wToPlot = np.array([weights[:, h] for h in hToPlotIndices])
    viz_rf(weights=wToPlot.reshape((image_size[0], image_size[1], -1)),
           it=epochId, grid=[2, int(numToPlot/2)])


def energyAvg(visible, hidden, weights, biasV, biasH, matrixOps=True):
    if matrixOps:
        term1 = np.sum(biasV * visible, axis=1)
        term2 = np.sum(biasH * hidden, axis=1)
        vExpanded = np.expand_dims(visible, axis=2)
        hExpanded = np.expand_dims(hidden, axis=1)
        # print("#####")
        # print(vExpanded.shape, hExpanded.shape)
        mulmul = tf.matmul(vExpanded, hExpanded)
        # print(mulmul.shape)
        wExpanded = np.expand_dims(weights, axis=0)
        product = mulmul * wExpanded
        sumDataPoints = np.sum(np.sum(product, axis=2), axis=1)
        term3 = sumDataPoints
        # print(term1.shape, term2.shape, term3.shape)
        return np.mean(- term1 - term2 - term3)
    else:
        energySum = 0
        N = len(visible)
        for i in range(N):
            energySum += energy(visible[i], hidden[i], weights, biasV, biasH)
        return energySum / N


def energy(visible, hidden, weights, biasV, biasH):
    term1 = np.sum(biasV * visible)
    term2 = np.sum(biasH * hidden)
    vis = np.expand_dims(visible, axis=1)
    hid = np.expand_dims(hidden, axis=0)
    # print(vis.shape, hid.shape)
    term3 = np.sum((vis @ hid) * weights)
    return - term1 - term2 - term3
