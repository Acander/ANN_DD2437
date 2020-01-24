from Lab1 import Activations, Losses
import numpy as np


class DenseLayer:

    def __init__(self, inSize, outSize, activation):
        self.weights = self._initLayerWeights(inSize, outSize)
        self.activation = activation
        self.cacheInput = 0
        self.cachePreActivation = 0
        self.cachePostActivation = 0

    def _initLayerWeights(self, sizeIn, sizeOut):
        return np.random.normal(0, 0.5, (sizeOut, sizeIn))

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
        x = x.T
        for l in self.layers:
            x = l.forward(x)
        return x

    def _backprop(self, labels):
        gradientProduct = self.loss.derivative(self.layers[-1].cachePostActivation, labels)
        for i, l in enumerate(reversed(self.layers)):
            gradientProduct *= l.activation.derivative(l.cachePreActivation)
            weightGrad = np.matmul(gradientProduct, l.cacheInput.T)

            l.weights -= weightGrad * self.lr
            gradientProduct = np.matmul(l.weights.T, gradientProduct)


    def fit(self, x, labels):
        y = self.forwardPass(x)
        self._backprop(labels)
        return self.loss.forward(y, labels)

    def __str__(self):
        return str([l.weights.shape for l in self.layers])


np.random.seed(42)
layers = [
    DenseLayer(2, 4, Activations.Sigmoid()),
    DenseLayer(4, 1, Activations.Sigmoid()),
]

myNet = FeedForwardNet(layers, Losses.MSE(), 0.004)
print("Network:", myNet)

inData = np.array([[2, 1], [1, 2]])
labels = np.array([[1], [0]])

for i in range(100000):
    out = myNet.forwardPass(inData)
    loss = myNet.fit(inData, labels)
    print(loss)
