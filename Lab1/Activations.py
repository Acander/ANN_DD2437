import numpy as np


class Activation:

    def forward(self, x):
        return x

    def derivative(self, x):
        return x


class ReLu(Activation):
    def __init__(self):
        self.f = np.vectorize(lambda x: x if x > 0 else 0)
        self.d = np.vectorize(lambda x: 1 if x > 0 else 0)

    def forward(self, x):
        return self.f(x)

    def derivative(self, x):
        return self.d(x)


class Linear(Activation):

    def forward(self, x):
        return x

    def derivative(self, x):
        return np.ones(x.shape)


class Sigmoid(Activation):

    def __init__(self):
        self.f = np.vectorize(lambda x: 1 / (1 + np.exp(-x)))
        self.forwardCache = 0

    def forward(self, x):
        self.forwardCache = self.f(x)
        return self.forwardCache

    def derivative(self, x):
        return self.forwardCache * (1 - self.forwardCache)
