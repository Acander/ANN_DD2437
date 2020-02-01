import numpy as np
import numpy.random as r


def generateData(box=False, shuffle=True):
    xTrain = np.arange(0, 2 * np.pi, 0.1)
    xTest = np.arange(0.05, 2 * np.pi, 0.1)
    if shuffle:
        r.shuffle(xTrain)
        r.shuffle(xTest)

    yTrain = np.sin(xTrain)
    yTest = np.sin(xTest)

    if box:
        yTrain[np.where(yTrain >= 0)] = 1
        yTrain[np.where(yTrain < 0)] = -1
        yTest[np.where(yTest >= 0)] = 1
        yTest[np.where(yTest < 0)] = -1

    N = len(xTrain)

    return np.reshape(xTrain, (N, 1)), \
           np.reshape(yTrain, (N, 1)), \
           np.reshape(xTest, (N, 1)), \
           np.reshape(yTest, (N, 1))


def evaluateModel(model, X, Y, residualError=True):
    predictions = model.predict(X)
    error = np.abs(Y - predictions) if residualError else np.square(Y - predictions)
    return np.mean(error)


if __name__ == '__main__':
    print(generateData(True))
