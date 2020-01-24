from Lab1 import Activations, Losses
import numpy as np


class DenseLayer:

    def __init__(self, inSize, outSize, activation):
        self.weights = self._initLayerWeights(inSize, outSize)
        self.activation = activation
        self.cacheInput = 0
        self.cachePreActivation = 0
        self.cachePostActivation = 0

    def _initLayerWeights(self, sizeIn, sizeOut):  # +
        return np.random.normal(0, 0.5, (sizeOut, sizeIn + 1))

    def forward(self, x):
        self.cacheInput = x
        self.cachePreActivation = np.matmul(self.weights, x)
        self.cachePostActivation = self.activation.forward(self.cachePreActivation)
        return self.cachePostActivation


class FeedForwardNet:

    def __init__(self, layers, loss, learningRate):
        self.lr = learningRate
        self.layers = layers
        self.loss = loss
        self.cache = []

    def forwardPass(self, x):
        for l in self.layers:
            newX = np.vstack([x, np.ones([1, x.shape[1]])])
            x = l.forward(newX)
        return x

    def _backprop(self, labels):
        gradientProduct = self.loss.derivative(self.layers[-1].cachePostActivation, labels)
        print("Grad Shape", gradientProduct.shape)
        for i, l in enumerate(reversed(self.layers)):
            print("\n", i)
            gradientProduct *= l.activation.derivative(l.cachePreActivation)
            print("Grad Shape", gradientProduct.shape)
            print("Cache Input", l.cacheInput.T.shape)
            weightGrad = np.matmul(gradientProduct, l.cacheInput.T)
            print("Weights shape", l.weights.shape)
            print("Weight grad", weightGrad.shape)
            l.weights -= weightGrad * self.lr

            gradientProduct = np.matmul(l.weights.T, gradientProduct)
            # print("Gradient prod", gradientProduct.shape)
            gradientProduct = gradientProduct[:-1]

    def fit(self, x, labels):
        y = self.forwardPass(x)
        self._backprop(labels)
        return self.loss.forward(y, labels)

    def __str__(self):
        return str([l.weights.T.shape for l in self.layers])


# np.random.seed(42)
layers = [
    DenseLayer(2, 2, Activations.Sigmoid()),
    DenseLayer(2, 1, Activations.Linear()),
]

myNet = FeedForwardNet(layers, Losses.MSE(), 0.002)
print("Network:", myNet)

inData = np.array([[1, 0], [0, 1], [0, 0], [1, 1]]).T
labels = np.array([[1], [0], [0], [1]]).T

for i in range(100000):
    loss = myNet.fit(inData, labels)
    print(loss)
