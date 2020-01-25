from Lab1.NeuralNetwork import FeedForwardNet, DenseLayer
from Lab1 import Activations, Losses, Utils
import numpy as np

layers = [
    DenseLayer(2, 5, Activations.ReLu()),
    DenseLayer(5, 2, Activations.ReLu()),
    DenseLayer(2, 1, Activations.Sigmoid()),
]
'''
myNet = FeedForwardNet(layers, Losses.MSE(), learningRate=0.005)
print("Network:", myNet)

inData = np.array([[4, 3], [2, 1]]).T
labels = np.array([[0], [1]]).T
print(inData)

for i in range(100000):
    loss = myNet.fit(inData, labels)
    print("\n", i, "Loss", loss)
    out = myNet.forwardPass(inData)
    print("Output", out)
'''

