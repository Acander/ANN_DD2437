import numpy as np
import os


def generateModel():
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Removes the use of GPU
    from tensorflow.keras import Model, layers

    inLayer = layers.Input((5,))
    d1 = layers.Dense(25, 'relu')(inLayer)
    d2 = layers.Dense(25, 'relu')(d1)
    d3 = layers.Dense(1, 'linear')(d2)
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
        labels.append([prevData[t + 5 + delay]])

    return np.array(inData), np.array(labels)


inData, labels = generateGlassData(10000)
print("InData:", inData.shape, " Labels:", labels.shape)
model = generateModel()
model.compile('adam', 'mse')
model.summary()

model.fit(inData, labels, batch_size=32, epochs=100, shuffle=True)
