import numpy as np


class MSE:

    def forward(self, y, label):
        return np.mean((y - label) ** 2)

    def derivative(self, y, label):
        loss = 2 * (y - label)
        return np.mean(loss, axis=0).reshape((1, y.shape[1]))
