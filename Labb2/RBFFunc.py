import numpy as np

SIGMA = 0.04
SIGMA22 = 2 * SIGMA * SIGMA


def RBF(distSquared):
    return np.exp(- distSquared / SIGMA22)
