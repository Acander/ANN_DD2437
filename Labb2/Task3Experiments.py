import os

from Lab1.NeuralNetwork import DenseLayer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Removes the use of GPU

from Labb2 import DataHandler, RBFFunc, RBFNet, TFNet
from Lab1 import NeuralNetwork, Losses, Activations
import numpy as np

if __name__ == '__main__':
    import tensorflow as tf
    import matplotlib.pyplot as plt

    '''
    data = {'1': 0.5700165, '2': 0.5699499, '3': 0.50881517, '4': 0.23135496, '5': 0.09188146, '6': 0.09100131,
            '7': 0.07595688,
            '8': 0.065666445, '9': 0.059209283, '10': 0.053510815,
            '11': 0.053510815, '12': 0.039748915, '13': 0.0350893, '14': 0.030781662, '15': 0.027155636,
            '16': 0.023469623, '17': 0.020750074, '18': 0.018196093, '19': 0.01592614, '20': 0.014122872,
            '21': 0.01288203, '22': 0.012161781, '32': 0.0035392188, '42': 0.0021866895, '52': 0.0010628038,
            '62': 0.0009942213
            }

    newData = {int(k): data[k] for k in data}

    for j in [22, 32, 42, 52]:
        for i in range(j + 1, j + 10):
            newData[i] = data[str(j)] + (data[str(j + 10)] - data[str(j)]) / 10 * (i - j)

    print(newData)
    keys = list(newData.keys())
    keys.sort()
    print(keys)
    plt.plot(keys, [newData[k] for k in keys], color='red', linewidth=3)
    plt.title = "Residual Error on test set"
    plt.xlabel('Number of hidden nodes')
    plt.ylabel('Residual Error')
    plt.show()
    '''
    np.random.seed(43)
    _, _, cleanTestX, cleanTestY = DataHandler.generateData(noiseVariance=0, box=False)
    cleanTestX = tf.convert_to_tensor(cleanTestX, dtype=tf.float32)
    cleanTestY = tf.convert_to_tensor(cleanTestY, dtype=tf.float32)

    trainX, trainY, testX, testY = DataHandler.generateData(noiseVariance=0, box=False)
    trainX = tf.convert_to_tensor(trainX, dtype=tf.float32)
    trainY = tf.convert_to_tensor(trainY, dtype=tf.float32)
    testX = tf.convert_to_tensor(testX, dtype=tf.float32)
    testY = tf.convert_to_tensor(testY, dtype=tf.float32)
    results = {}

    numberOfPoints = 20
    steps = 13000
    #points = np.array([[np.pi / (numberOfPoints / 2) * i] for i in range(numberOfPoints)])
    #print(points.shape)

    # for r in [0.0001, 0.0005, 0.001, 0.002, 0.003, 0.005]:
    #for r in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2, 1.4, 1.6]:
    for r in range(1, 60):
        points = np.random.uniform(0, np.max(trainX), (r, 1))
        print(points.shape)

        #myModel = NeuralNetwork.FeedForwardNet(layers, Losses.MSE(), 0.001)

        myModel = TFNet.RadialBasisFunctionNetwork(1, 1, r, points, 1, lr=0.001)
        results[r] = []
        for i in range(steps):
            '''
            randomOrder = np.random.choice(np.arange(len(trainX)), len(trainX), replace=False)
            batchLoss = []
            for p in randomOrder:
                loss = myModel.fit([trainX[p]], [trainY[p]])
                batchLoss.append(loss)
            if (i % 10 == 0):
                print(np.mean(batchLoss), myModel.lr, i, "/", steps)

            results[r].append(np.mean(batchLoss))
            '''
            loss = myModel.fit(trainX, trainY)
            #results[r].append(loss)
            if (i % 1000 == 0):
                print(loss, myModel.lr, i, "/", steps)

        testPred = myModel.predict(np.array(testX))
        error = tf.losses.MeanAbsoluteError()(testY, testPred)
        print("Test Error:", error, r)

        testPred = myModel.predict(np.array(cleanTestX))
        error = tf.losses.MeanAbsoluteError()(cleanTestY, testPred)
        print("Clean Test Error:", error, r)
        results[r] = error

    import pickle

    with open("Lr_Results_Seq_Sin.pkl", 'wb') as fp:
        pickle.dump(results, fp)
    print(results)
    '''
    '''
