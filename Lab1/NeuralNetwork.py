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
        return np.random.normal(0, 0.5, (sizeIn, sizeOut))

    def forward(self, x):
        self.cacheInput = x
        self.cachePreActivation = np.matmul(x, self.weights)
        self.cachePostActivation = self.activation.forward(self.cachePreActivation)
        return self.cachePostActivation


class FeedForwardNet:

    def __init__(self, layers, loss, learningRate):
        self.lr = learningRate
        self.layers = layers
        self.loss = loss
        self.cache = []

    # Ja
    def forwardPass(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x


    def _backprop(self, labels):
        gradientProduct = self.loss.derivative(self.layers[-1].cachePostActivation, labels)
        for i, l in enumerate(reversed(self.layers)):
            print("\n", i)
            print("Gradient Product", gradientProduct.shape)
            print("Activation grad:", l.activation.derivative(l.cachePreActivation).shape)
            gradientProduct *= l.activation.derivative(l.cachePreActivation)
            print("Gradient Product", gradientProduct.shape)
            print("Cache input", l.cacheInput.shape)
            weightGrad = gradientProduct * l.cacheInput
            print("Weight grad", weightGrad.shape)

            l.weights -= weightGrad.reshape(l.weights.shape)

    def fit(self, x, labels):
        y = self.forwardPass(x)
        gradient = self._backprop(labels)
        for i, g in reversed(gradient):
            self.layers[i][0] -= g * self.lr

        return self.loss.forward(y, labels)

    def __str__(self):
        return str([l.weights.shape for l in self.layers])


np.random.seed(42)
layers = [
    DenseLayer(2, 3, Activations.ReLu()),
    DenseLayer(3, 1, Activations.ReLu())
]

myNet = FeedForwardNet(layers, Losses.MSE(), 0.001)
print("Network:", myNet)

inData = np.array([[1, 1]])
labels = np.array([[4]])
for i in range(5):
    out = myNet.forwardPass(inData)
    loss = myNet.fit(inData, labels)
    print(loss)
