from Labb2 import DataHandler, RBFFunc, RBFNet
import numpy as np

if __name__ == '__main__':
    numberOfPoints = 200
    #points = np.array([[np.pi / (numberOfPoints/2) * i] for i in range(numberOfPoints)])
    points = np.random.uniform(0, 2*np.pi, (numberOfPoints, 1))
    print(points.shape)

    trainX, trainY, testX, testY = DataHandler.generateData()
    print(trainX.shape)

    myModel = RBFNet.RadialBasisFunctionNetwork(1, 1, numberOfPoints, points, RBFFunc.RBF, lr=0.005)
    for i in range(1000000):
        #for x, y in zip(trainX[:1], trainY[:1]):
        #    print(myModel.fit(np.array([x]), np.array([y])))
        print(myModel.fit(trainX, trainY))

