from Lab1 import Utils
import numpy as np
import os
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Removes the use of GPU


def generateModel(reg):
    from tensorflow.keras import Model, layers, regularizers
    # reg = 0.005
    inLayer = layers.Input((5,))
    d1 = layers.Dense(8, 'tanh', use_bias=True,
                      kernel_regularizer=regularizers.l2(reg))(inLayer)
    d2 = layers.Dense(100, 'tanh', use_bias=True,
                      kernel_regularizer=regularizers.l2(reg))(d1)
    d3 = layers.Dense(1, 'linear', use_bias=True, kernel_regularizer=regularizers.l2(reg))(d2)

    return Model(inLayer, d3)


def _mackeyGlassData(previousValues, beta=0.2, gamma=0.1, n=10, delay=25):
    return previousValues[-1] * (1 - gamma) + beta * previousValues[-delay] / (1 + previousValues[-delay] ** n)


def generateGlassData(numberOfsamples, min=301, max=1500, history=(-20, -15, -10, -5, 0), beta=0.2, gamma=0.1,
                      n=10, delay=25):
    prevData = [0 for _ in range(delay - 1)] + [1.5]
    timeStamps = np.random.randint(min, max, numberOfsamples)
    for i in range(np.max(timeStamps) + delay + 5):
        newValue = _mackeyGlassData(prevData, beta, gamma, n, delay)
        prevData.append(newValue)

    inData = []
    labels = []
    for t in timeStamps:
        inData.append([prevData[t + h + delay] for h in history])
        # inData.append([prevData[t + delay] for h in history])
        labels.append([prevData[t + 5 + delay]])

    return np.array(inData), np.array(labels)


from tensorflow.keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def weightHistogram(model):
    weights1, biases1 = model.layers[1].get_weights()
    weights2, biases2 = model.layers[2].get_weights()
    print("Shape:", weights1.shape)
    weights = np.concatenate([weights1.flatten(), weights2.flatten()]).flatten()
    print(weights)
    plt.style.use('ggplot')
    plt.hist(weights, bins=50)
    plt.xlabel("weight value")
    plt.ylabel("frequency")
    plt.show()



samples = np.random.randint(301, 1500, 1200).tolist()
from Lab1 import StolenData

trainSize = 400  # Training samples
noise = 0.09  # Noise STD

inData, labels = StolenData.getData(1200, 300, 4500, samples)  # generateGlassData(1200)
print("InData:", inData.shape, " Labels:", labels.shape)

# Adding of noise
inData += np.random.normal(0, noise, inData.shape)
labels += np.random.normal(0, noise, labels.shape)

inTest, labelTest = inData[-200:], labels[-200:]
print("InTest:", inTest.shape, "LabelTest:", labelTest.shape)
inTrain, labelTrain, inEval, labelEval, = inData[:trainSize], labels[:trainSize], \
                                          inData[trainSize:1000], labels[trainSize:1000]
print("InTrain:", inTrain.shape, "TrainLabel:", labelTrain.shape)
print("InEval:", inEval.shape, "EvalLabel:", labelEval.shape)

results = {}
import time
tStart = time.time()
for h in [0, 0.01]:  # , 0.0001, 0.0005, 0.001, 0.0015]:
    model = generateModel(6)
    model.compile('adam', 'mse', metrics=[root_mean_squared_error])

    history = model.fit(inTrain, labelTrain, epochs=800, validation_data=[inEval, labelEval])
    loss, evalLoss = history.history['loss'][5:], history.history['val_loss'][5:]

    results[h] = model.evaluate(inTest, labelTest)
    weightHistogram(model)

print("Time:", time.time() - tStart)

print(results)
