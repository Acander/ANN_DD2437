from Labb2.RBFFunc import RBF
import tensorflow as tf
import numpy as np


class RadialBasisFunctionNetwork(tf.keras.Model):

    def distFunc(self, x):
        return tf.exp(-x / self.radius)

    def __init__(self, inputSize, outSize, hiddenSize, hiddenCentroids, radius, lr=0.05, l1Dist=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.outWeights = tf.convert_to_tensor(np.random.normal(-0.05, 0.05, (hiddenSize, outSize)), dtype=tf.float32)
        self.outWeights = tf.Variable(self.outWeights)
        self.centroids = tf.constant(hiddenCentroids, dtype=tf.float32)
        self.numHidden = hiddenSize
        self.l1Dist = l1Dist
        self.radius = 2*radius*radius
        self.previousHiddenValues = "CACHED FROM PREDICTION"
        self.lr = tf.convert_to_tensor(lr, dtype=tf.float32)
        self.optimizer = tf.optimizers.Adam(lr=lr)
        assert hiddenCentroids.shape[0] == hiddenSize and hiddenCentroids.shape[1] == inputSize

    def predictDist(self, inputs):
        inputs = tf.expand_dims(inputs, axis=1)
        s = inputs.shape
        inputs = tf.broadcast_to(inputs, (s[0], self.numHidden, s[2]))
        # print(inputs.shape)
        # print(self.centroids)
        centroids = tf.expand_dims(self.centroids, axis=0)
        # print(centroids)
        deltaDist = inputs - centroids
        # print(deltaDist)
        deltaDist **= 2
        # print("Delta Dist", deltaDist.shape)
        deltaDist = tf.reduce_sum(deltaDist, axis=-1)
        # print(deltaDist)
        return deltaDist

    def predict(self, inputs):
        # inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        nodeDists = self.predictDist(inputs)
        kernelDist = self.distFunc(nodeDists)

        # print(kernelDist)
        # print(self.outWeights)
        return tf.matmul(kernelDist, self.outWeights)

    def fit(self, inputs, labels):
        # labels = tf.convert_to_tensor(labels, dtype=tf.float32)
        with tf.GradientTape() as tape:
            #tape.watch(self.outWeights)
            pred = self.predict(inputs)
            # print(pred, labels)
            loss = tf.losses.MeanSquaredError()(labels, pred)  # Allows for batch
            # print(loss)

        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        #self.outWeights -= grad * self.lr
        #self.outWeights.assign_sub(grad[0]*self.lr)
        #tf.Variable.assign_sub(self.outWeights, grad*self.lr)

        return loss

