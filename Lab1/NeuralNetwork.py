import Activations, Losses
import numpy as np


class DenseLayer:

    def __init__(self, inSize, outSize, activation):
        self.weights = self._initLayerWeights(inSize, outSize)
        self.activation = activation
        self.cacheInput = 0
        self.cachePreActivation = 0
        self.cachePostActivation = 0

    def _initLayerWeights(self, sizeIn, sizeOut):  # +1 is for the Bias that will come in
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
            xWithBias = np.vstack([x, np.ones([1, x.shape[1]])])
            x = l.forward(xWithBias)
        return x

    def _backprop(self, labels):
        gradientProduct = self.loss.derivative(self.layers[-1].cachePostActivation, labels)
        for l in reversed(self.layers):
            gradientProduct *= l.activation.derivative(l.cachePreActivation)
            weightGrad = np.matmul(gradientProduct, l.cacheInput.T)
            l.weights -= weightGrad * self.lr

            gradientProduct = np.matmul(l.weights.T, gradientProduct)
            gradientProduct = gradientProduct[:-1]  # In the backprop the bias is not to be included further

    def fit(self, x, labels, batchSize=-1):
        if (batchSize == -1):
            batchSize = x.shape[1]

        losses = []
        preOut = self.forwardPass(x).copy()
        for i in range(0, x.shape[1], batchSize):
            batchX = x[:, i:i + batchSize]
            batchLabel = labels[:, i:i + batchSize]

            y = self.forwardPass(batchX)
            self._backprop(batchLabel)
            losses.append(self.loss.forward(y, batchLabel))

        return np.mean(self.loss.forward(preOut, labels))

    def __str__(self):
        return str([(l.weights.T.shape[0] - 1,) + l.weights.T.shape[1:] for l in self.layers])
