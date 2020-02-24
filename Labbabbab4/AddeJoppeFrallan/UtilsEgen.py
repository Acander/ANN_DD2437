import numpy as np
import tensorflow as tf


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
