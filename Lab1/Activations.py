import numpy as np


class Activation:

    def forward(self, x):
        return x

    def derivative(self, x):
        return x


class ReLu(Activation):
    def __init__(self):
        self.f = np.vectorize(lambda x: x if x > 0 else 0)

    def forward(self, x):
        return self.f(x)

    def derivative(self, x):
        return self.f(x)


class Linear(Activation):

    def __init__(self):
        pass

    def forward(self, x):
        return x

    def derivative(self, x):
        return x
