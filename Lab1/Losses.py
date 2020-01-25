import numpy as np


class MSE:

    def forward(self, y, label):
        return np.mean((y - label) ** 2)

    def derivative(self, y, label):
        loss = 2 * (y - label)
        return loss
