import numpy as np


class MSE:

    def forward(self, y, label):
        return np.mean((y - label) ** 2)

    def derivative(self, y, label):
        return 2 * np.mean(y - label, axis=0).reshape((1, y.shape[1]))
