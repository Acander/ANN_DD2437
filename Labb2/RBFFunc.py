import numpy as np


def RBF(distSquared, sigma):
    return np.exp(- distSquared / (2 * sigma * sigma))
