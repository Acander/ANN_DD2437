import os

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
from Labbabbab4.AddeJoppeFrallan import UtilsEgen

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from Labbabbab4.codeAlaPawel.util import *
import tensorflow as tf
import time, json


class RestrictedBoltzmannMachine(tf.keras.Model):

    def __init__(self, ndim_visible, ndim_hidden, is_bottom=False, image_size=(28, 28), is_top=False, n_labels=10,
                 batch_size=10, learning_rate=0.1, name="", *args, **kwargs):

        """
        Args:
          ndim_visible: Number of units in visible layer.
          ndim_hidden: Number of units in hidden layer.
          is_bottom: True only if this rbm is at the bottom of the stack in a deep belief net. Used to interpret visible layer as image data with dimensions "image_size".
          image_size: Image dimension for visible layer.
          is_top: True only if this rbm is at the top of stack in deep beleif net. Used to interpret visible layer as concatenated with "n_label" unit of label data at the end. 
          n_label: Number of label categories.
          batch_size: Size of mini-batch.
        """

        super().__init__(*args, **kwargs)
        self.ndim_visible = ndim_visible
        self.ndim_hidden = ndim_hidden
        self.batch_size = batch_size
        self.rbName = name

        self.is_bottom = is_bottom
        self.is_top = is_top
        if is_bottom:
            self.image_size = image_size
        if is_top:
            self.n_labels = 10

        f = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)
        g = lambda x, n: tf.Variable(f(x), name=n)
        self.delta_bias_v = g(np.zeros(self.ndim_visible), "delta_bias_v")
        self.delta_bias_h = g(np.zeros(self.ndim_hidden), "delta_bias_h")
        self.delta_weight_vh = g(np.zeros((self.ndim_visible, self.ndim_hidden)), "delta_weight_vh")

        self.bias_v = g(np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible)), "bias_v")
        self.bias_h = g(np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_hidden)), "bias_h")
        self.weight_vh = g(np.random.normal(loc=0.0, scale=0.01, size=(self.ndim_visible, self.ndim_hidden)),
                           "weight_vh")
        self.weight_v_to_h = g(np.zeros((self.ndim_visible, self.ndim_hidden)),
                               "weight_v_to_h")
        self.weight_h_to_v = g(np.zeros((self.ndim_visible, self.ndim_hidden)), "weight_h_to_v")
        self.hasUntwinedWeights = tf.Variable(False, name="HasUntwined")

        self.delta_weight_v_to_h = g(np.zeros((self.ndim_visible, self.ndim_hidden)), "delta_weight_v_to_h")
        self.delta_weight_h_to_v = g(np.zeros((self.ndim_visible, self.ndim_hidden)), "delta_weight_h_to_v")

        self.learning_rate = g(learning_rate, "learning_rate")
        self.momentum = g(0.7, 'momentum')

        self.testLosses = []
        self.trainLosses = []

        self.print_period = 5000
        self.rf = {  # receptive-fields. Only applicable when visible layer is input data
            "period": 5000,  # iteration period to visualize
            "grid": [5, 5],  # size of the grid
            "ids": np.random.randint(0, self.ndim_hidden, 25)  # pick some random hidden units
        }
        self.allVariables = {
            'delta_bias_v': self.delta_bias_v, 'delta_bias_h': self.delta_bias_h,
            'delta_weight_vh': self.delta_weight_vh,
            'bias_v': self.bias_v, 'bias_h': self.bias_h, 'weight_vh': self.weight_vh,
            'weight_v_to_h': self.weight_v_to_h, 'weight_h_to_v': self.weight_h_to_v,
            'delta_weight_v_to_h': self.delta_weight_v_to_h, 'delta_weight_h_to_v': self.delta_weight_h_to_v,
        }

    def divideIntoParts(self, data, partSize=1000):
        f = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)
        numberOfParts = int(np.ceil(len(data)) / partSize)
        return [f(data[i * partSize:(i + 1) * partSize]) for i in range(0, numberOfParts)]

    def cd1(self, visible_trainset, testSet=(), numEpochs=10000):

        """Contrastive Divergence with k=1 full alternating Gibbs sampling

        Args:
          visible_trainset: training data for this rbm, shape is (size of training set, size of visible layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """
        print("learning CD1")

        trainingSetParts = self.divideIntoParts(visible_trainset, 1000)
        if (len(testSet) != 0):
            testSetParts = self.divideIntoParts(testSet, 1000)
        print("Number of parts:", len(trainingSetParts))

        if (len(testSet) != 0):
            self.testLosses.append(UtilsEgen.meanReconstLossOnParts(self, testSetParts))
        self.trainLosses.append(UtilsEgen.meanReconstLossOnParts(self, trainingSetParts))

        for epoch in range(numEpochs):
            print("Epoch: {}/{}".format(epoch, numEpochs))
            visible_trainset = tf.random.shuffle(visible_trainset)
            epochStartTime = time.time()

            for p in trainingSetParts:
                numIterationsInBatch = int(np.ceil(len(p)) / self.batch_size)
                for b in range(0, numIterationsInBatch):
                    dataBatch = p[b * self.batch_size: (b + 1) * self.batch_size]
                    loss = self.forwardAndUpdate(dataBatch)
                    if (b == 0):
                        print(loss)

            # print("Recloss:", np.round(np.mean(epochLoss / numIterationsInBatch), decimals=3))
            print("Evaluating...")
            self.trainLosses.append([UtilsEgen.meanReconstLossOnParts(self, trainingSetParts)])
            if (len(testSet) != 0):
                self.testLosses.append([UtilsEgen.meanReconstLossOnParts(self, testSetParts)])
                print("Test Loss:", self.testLosses[-1])

            print("Training Loss:", self.trainLosses[-1])

            print("Epoch Time:", time.time() - epochStartTime)
            self.learning_rate = self.learning_rate * 0.99
            print("New Learning Rate", self.learning_rate)
            # UtilsEgen.plotWeights(self.weight_vh, epoch)

        '''
        with open("RunStatsLearning-{}.json".format(self.rbName), 'w') as fp:
            json.dump({'TrainingLoss': self.trainLosses, 'TestLoss': self.testLosses}, fp)
        '''

    # @tf.function
    def forwardAndUpdate(self, dataBatch):
        ph0, h0 = self.get_h_given_v(dataBatch)
        pv1, v1 = self.get_v_given_h(h0)
        ph1, h1 = self.get_h_given_v(v1)
        self.update_params(dataBatch, h0, v1, ph1)
        return UtilsEgen.meanReconstLoss(dataBatch, v1)

    def update_params(self, v_0, h_0, v_k, h_k):
        t = tf.matmul(tf.expand_dims(v_0, axis=2), tf.expand_dims(h_0, axis=1))
        t2 = tf.matmul(tf.expand_dims(v_k, axis=2), tf.expand_dims(h_k, axis=1))
        t = t - t2

        meanDeltaWeights = tf.reduce_mean(t, axis=0)
        meanDeltaBiasV = tf.reduce_mean(v_0 - v_k, axis=0)
        meanDeltaBiasH = tf.reduce_mean(h_0 - h_k, axis=0)

        # print(meanDeltaBiasV)
        lr = self.learning_rate
        self.delta_bias_v = self.delta_bias_v * self.momentum + (1 - self.momentum) * meanDeltaBiasV
        self.delta_bias_h = self.delta_bias_h * self.momentum + (1 - self.momentum) * meanDeltaBiasH
        self.delta_weight_vh = self.delta_weight_vh * self.momentum + (1 - self.momentum) * meanDeltaWeights

        self.bias_v.assign_add(self.delta_bias_v * lr)
        self.weight_vh.assign_add(self.delta_weight_vh * lr)
        self.bias_h.assign_add(self.delta_bias_h * lr)

    # @tf.function
    def get_h_given_v(self, batch):
        if (self.hasUntwinedWeights):
            res = tf.matmul(batch, self.weight_h_to_v) + self.bias_h
            probs = sigmoid(res)
            return probs, sample_binary(probs)
        else:
            res = tf.matmul(batch, self.weight_vh) + self.bias_h
            probs = sigmoid(res)
            return probs, sample_binary(probs)

    # @tf.function
    def get_v_given_h(self, hidden_minibatch):
        if (self.hasUntwinedWeights):
            res = tf.matmul(hidden_minibatch, tf.transpose(self.weight_h_to_v)) + self.bias_v
            probs = sigmoid(res)
            return probs, sample_binary(probs)
        else:
            res = tf.matmul(hidden_minibatch, tf.transpose(self.weight_vh)) + self.bias_v
            if self.is_top:
                N = len(hidden_minibatch)

                probsSM = tf.nn.softmax(res[:, -10:])
                probsRest = sigmoid(res[:, :-10])
                probs = tf.concat([probsRest, probsSM], axis=1)

                argMax = tf.reshape(tf.argmax(probsSM, axis=1), (1, N))
                samplesSM = tf.one_hot(argMax[0], 10)
                samplesRest = sample_binary(probsRest)
                samples = tf.concat([samplesRest, samplesSM], axis=1)
            else:
                probs = sigmoid(res)
                samples = sample_binary(probs)

            return probs, samples

    def untwine_weights(self):
        self.weight_v_to_h.assign(self.weight_vh)
        self.weight_h_to_v.assign(self.weight_vh)
        self.hasUntwinedWeights.assign(True)
        # self.weight_vh = None

    '''
    def get_h_given_v_dir(self, visible_minibatch):

        """Compute probabilities p(h|v) and activations h ~ p(h|v)

        Uses directed weight "weight_v_to_h" and bias "bias_h"
        
        Args: 
           visible_minibatch: shape is (size of mini-batch, size of visible layer)
        Returns:        
           tuple ( p(h|v) , h) 
           both are shaped (size of mini-batch, size of hidden layer)
        """

        assert self.weight_v_to_h is not None

        n_samples = visible_minibatch.shape[0]

        # [TODO TASK 4.2] perform same computation as the function 'get_h_given_v' but with directed connections (replace the zeros below) 

        return np.zeros((n_samples, self.ndim_hidden)), np.zeros((n_samples, self.ndim_hidden))

    def get_v_given_h_dir(self, hidden_minibatch):

        """Compute probabilities p(v|h) and activations v ~ p(v|h)

        Uses directed weight "weight_h_to_v" and bias "bias_v"
        
        Args: 
           hidden_minibatch: shape is (size of mini-batch, size of hidden layer)
        Returns:        
           tuple ( p(v|h) , v) 
           both are shaped (size of mini-batch, size of visible layer)
        """

        assert self.weight_h_to_v is not None

        n_samples = hidden_minibatch.shape[0]

        if self.is_top:

            """
            Here visible layer has both data and labels. Compute total input for each unit (identical for both cases), \ 
            and split into two parts, something like support[:, :-self.n_labels] and support[:, -self.n_labels:]. \
            Then, for both parts, use the appropriate activation function to get probabilities and a sampling method \
            to get activities. The probabilities as well as activities can then be concatenated back into a normal visible layer.
            """

            # [TODO TASK 4.2] Note that even though this function performs same computation as 'get_v_given_h' but with directed connections,
            # this case should never be executed : when the RBM is a part of a DBN and is at the top, it will have not have directed connections.
            # Appropriate code here is to raise an error (replace pass below)

            pass

        else:

            # [TODO TASK 4.2] performs same computaton as the function 'get_v_given_h' but with directed connections (replace the pass and zeros below)             

            pass

        return np.zeros((n_samples, self.ndim_visible)), np.zeros((n_samples, self.ndim_visible))
    '''

    def update_generate_params(self, inps, trgs, preds):
        """Update generative weight "weight_h_to_v" and bias "bias_v"
        
        Args:
           inps: activities or probabilities of input unit -> X
           trgs: activities or probabilities of output unit (target) -> Y
           preds: activities or probabilities of output unit (prediction) -> Ỹ
           all args have shape (size of mini-batch, size of respective layer)
        """

        # print("X:", inps.shape)
        # print("Y:", trgs.shape)
        # print("Y2:", preds.shape)
        # print("Weights:", self.weight_h_to_v.shape)
        deltaActv = trgs - preds
        # print("DeltaActv:", deltaActv.shape)
        deltaW = tf.matmul(tf.expand_dims(deltaActv, axis=2), tf.expand_dims(inps, axis=1))
        deltaW = tf.reduce_mean(deltaW, axis=0)
        # print("DeltaW:", deltaW.shape)

        # print(self.delta_bias_v.shape)
        lr = self.learning_rate
        self.delta_weight_h_to_v.assign_add(deltaW * lr)
        self.delta_bias_v.assign_add(tf.reduce_mean(deltaActv, axis=0) * lr)

        self.weight_h_to_v = self.weight_h_to_v + self.delta_weight_h_to_v
        self.bias_v = self.bias_v + self.delta_bias_v

    def calcDelta(self, inps, trgs, preds):
        deltaActv = trgs - preds
        deltaW = tf.matmul(tf.expand_dims(deltaActv, axis=2), tf.expand_dims(inps, axis=1))
        return tf.reduce_mean(deltaW, axis=0), tf.reduce_mean(deltaActv, axis=0)

    def applyRecognizeDeltas(self, deltas):
        deltaW = [d[0] for d in deltas]
        deltaActvs = [d[1] for d in deltas]
        deltaW = tf.reduce_mean(tf.convert_to_tensor(deltaW), axis=0)
        deltaActvs = tf.reduce_mean(tf.convert_to_tensor(deltaActvs), axis=0)

        lr = self.learning_rate
        self.delta_weight_v_to_h.assign_add(tf.transpose(deltaW) * lr)
        self.delta_bias_h.assign_add(deltaActvs * lr)

        self.weight_v_to_h.assign_add(self.delta_weight_v_to_h * lr)
        self.bias_h = self.bias_h + self.delta_bias_h * lr

    def applyGenerativeDeltas(self, deltas):
        deltaW = [d[0] for d in deltas]
        deltaActvs = [d[1] for d in deltas]
        deltaW = tf.reduce_mean(tf.convert_to_tensor(deltaW), axis=0)
        deltaActvs = tf.reduce_mean(tf.convert_to_tensor(deltaActvs), axis=0)

        lr = self.learning_rate
        #lr = 0.1
        self.delta_weight_h_to_v.assign_add(deltaW * lr)
        self.delta_bias_v.assign_add(deltaActvs * lr)

        self.weight_h_to_v = self.weight_h_to_v + self.delta_weight_h_to_v
        self.bias_v = self.bias_v + self.delta_bias_v

    def update_recognize_params(self, inps, trgs, preds):

        """Update recognition weight "weight_v_to_h" and bias "bias_h"
        
        Args:
           inps: activities or probabilities of input unit
           trgs: activities or probabilities of output unit (target)
           preds: activities or probabilities of output unit (prediction)
           all args have shape (size of mini-batch, size of respective layer)
        """

        # [TODO TASK 4.3] find the gradients from the arguments (replace the 0s below) and update the weight and bias parameters.

        # print("X:", inps.shape)
        # print("Y:", trgs.shape)
        # print("Y2:", preds.shape)
        # print("Weights:", self.weight_h_to_v.shape)
        deltaActv = trgs - preds
        # print("DeltaActv:", deltaActv.shape)
        deltaW = tf.matmul(tf.expand_dims(deltaActv, axis=2), tf.expand_dims(inps, axis=1))
        deltaW = tf.reduce_mean(deltaW, axis=0)
        # print("DeltaW:", deltaW.shape)

        # lr = self.learning_rate
        lr = 0.01
        # print(lr)
        self.delta_weight_v_to_h.assign_add(tf.transpose(deltaW) * lr)
        self.delta_bias_h.assign_add(tf.reduce_mean(deltaActv, axis=0) * lr)

        self.weight_v_to_h.assign_add(self.delta_weight_v_to_h * lr)
        self.bias_h = self.bias_h + self.delta_bias_h * lr

    def getWeightsInNumpyByName(self, name):
        for v in self.variables:
            if (v.name[:-2] == name):
                return v.numpy()

    def fixDeltaWeights(self):
        f = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)
        g = lambda x, n: tf.Variable(f(x), name=n)
        self.delta_bias_v = g(np.zeros(self.ndim_visible), "delta_bias_v")
        self.delta_bias_h = g(np.zeros(self.ndim_hidden), "delta_bias_h")
        self.delta_weight_vh = g(np.zeros((self.ndim_visible, self.ndim_hidden)), "delta_weight_vh")
