from Labbabbab4.codeAlaPawel.util import *
from Labbabbab4.codeAlaPawel.rbm import RestrictedBoltzmannMachine
import tensorflow as tf


class DeepBeliefNet(tf.keras.Model):
    ''' 
    For more details : Hinton, Osindero, Teh (2006). A fast learning algorithm for deep belief nets. https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    network          : [top] <---> [pen] ---> [hid] ---> [vis] 
                               `-> [lbl] 
    lbl : label
    top : top
    pen : penultimate
    hid : hidden
    vis : visible
    '''

    def __init__(self, trainingStep, sizes, image_size=784, n_labels=10, batch_size=20, *args, **kwargs):

        """
        Args:
          sizes: Dictionary of layer names and dimensions
          image_size: Image dimension of data
          n_labels: Number of label categories
          batch_size: Size of mini-batch
        """
        super().__init__(*args, **kwargs)  # Super-Importanté

        '''
        '''
        self.rbm_stack = {
            'vis--hid': RestrictedBoltzmannMachine(ndim_visible=sizes["vis"], ndim_hidden=sizes["hid"],
                                                   is_bottom=True, image_size=image_size, batch_size=batch_size,
                                                   name="First"),
            'hid--pen': RestrictedBoltzmannMachine(ndim_visible=sizes["hid"], ndim_hidden=sizes["pen"],
                                                   batch_size=batch_size, name="Second"),
            'pen+lbl--top': RestrictedBoltzmannMachine(ndim_visible=sizes["pen"] + sizes["lbl"],
                                                       ndim_hidden=sizes["top"],
                                                       is_top=True, n_labels=n_labels, batch_size=batch_size,
                                                       name="Third")
        }
        self.trainingStep = trainingStep
        self.sizes = sizes
        self.image_size = image_size
        self.batch_size = batch_size

        self.n_gibbs_recog = 20
        self.n_gibbs_gener = 200
        self.n_gibbs_wakesleep = 5

        self.print_period = 2000

        self.loadStackWeights(self.trainingStep)

    def recognize(self, true_img, true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_img: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """

        f = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)
        n_samples = true_img.shape[0]
        vis = true_img  # visible layer gets the image data
        lbl = np.ones(true_lbl.shape) / 10.  # start the net by telling you know nothing about labels

        # [TODO TASK 4.2] fix the image data in the visible layer and drive the network bottom to top. In the top RBM, run alternating Gibbs sampling \
        # and read out the labels (replace pass below and 'predicted_lbl' to your predicted labels).
        # NOTE : inferring entire train/test set may require too much compute memory (depends on your system). In that case, divide into mini-batches.

        pv, v = self.propagateToFinalRBM(f(vis))
        v = tf.concat([v, lbl], axis=1)
        topRBM = self.rbm_stack['pen+lbl--top']

        for _ in range(self.n_gibbs_recog):
            ph, h = topRBM.get_h_given_v(v)
            pv, v = topRBM.get_v_given_h(h)

        # predicted_lbl = np.zeros(true_lbl.shape)
        lblNeurons = v[:, -10:].numpy()
        predicted_lbl = lblNeurons
        # predicted_lbl = np.argmax(lblNeurons, axis=1)
        print("accuracy = %.2f%%" % (100. * np.mean(np.argmax(predicted_lbl, axis=1) == np.argmax(true_lbl, axis=1))))

    # Generates images for all passed in, but shows only the one specified in argument. NICE
    def generate(self, true_lbl, name, imageToShow=0):

        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """

        # f = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)
        n_sample = true_lbl.shape[0]

        # records = []
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([])
        ax.set_yticks([])

        lbl = true_lbl

        initV = tf.random.uniform((n_sample, 784), minval=0, maxval=2, dtype=tf.dtypes.int32)
        print(lbl.shape)
        print(initV.shape)

        # [TODO TASK 4.2] fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From the top RBM, drive the network \ 
        # top to the bottom visible layer (replace 'vis' from random to your generated visible layer).

        topRBM = self.rbm_stack['pen+lbl--top']
        # v = f(initV)
        v = tf.dtypes.cast(initV, dtype=tf.float32)
        pv, v = self.propagateToFinalRBM(v)
        v = tf.concat([v, lbl], axis=1)

        images = []
        for i in range(self.n_gibbs_gener):
            ph, h = topRBM.get_h_given_v(v)
            pv, v = topRBM.get_v_given_h(h)

            print("Mean v in Gibbs sample:", np.mean(v))

            v = tf.concat([v[:, :-10], lbl], axis=1)
            '''
            for d in range(n_sample):
                for j in range(10):
                    v[d][-10 + j].assign(lbl[d][j])
            '''
            visP, vis = self.propagateFromFinalRBM(v[:, :-10])
            images.append(vis)

        print("Generating:", np.argmax(lbl[imageToShow]))
        # ANIMATION DID NOT WORK ON MY COMPUTER => YOLO
        for i in range(0, len(images), 20):
            imgToShow = images[i][imageToShow]
            plt.imshow(imgToShow.numpy().reshape((28, 28)), cmap="bwr", vmin=0, vmax=1, animated=True,
                       interpolation=None)
            plt.show()

        # anim = stitch_video(fig, records).save("%s.generate%d.mp4" % (name, np.argmax(true_lbl)))
        # plt.imshow(records[-1])

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):
        f = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)
        # vis_trainset = f(vis_trainset)
        # CD-1 training for vis--hid
        if (self.trainingStep > 0):
            print("Skipping Pretraining of RBM Layer-0")
        else:
            print("training vis--hid")
            self.rbm_stack['vis--hid'].cd1(vis_trainset, numEpochs=n_iterations)
            self.rbm_stack["vis--hid"].untwine_weights()
            self.rbm_stack['vis--hid'].save_weights("DBN-RBM-0-Weights")

        # CD-1 training for hid--pen
        if (self.trainingStep > 1):
            print("Skipping Pretraining of RBM Layer-1")
        else:
            print("training hid--pen")
            for i in range(n_iterations):
                data = self.rbm_stack['vis--hid'].get_h_given_v(f(vis_trainset))[1]
                self.rbm_stack['hid--pen'].cd1(data, numEpochs=1)

            self.rbm_stack["hid--pen"].untwine_weights()
            self.rbm_stack['hid--pen'].save_weights("DBN-RBM-1-Weights")

        """ 
        CD-1 training for pen+lbl--top 
        """
        self.preTrainFinalLayer(f(vis_trainset), lbl_trainset, n_iterations)
        self.rbm_stack["pen+lbl--top"].save_weights("DBN-RBM-2-Weights")

    def propagateToFinalRBM(self, inData):
        ph0, h0 = self.rbm_stack['vis--hid'].get_h_given_v(inData)
        return self.rbm_stack['hid--pen'].get_h_given_v(h0)

    def propagateFromFinalRBM(self, data):
        pv1, v1 = self.rbm_stack['hid--pen'].get_v_given_h(data)
        return self.rbm_stack['vis--hid'].get_v_given_h(v1)

    def preTrainFinalLayer(self, inData, labels, numIterations):
        topRBM = self.rbm_stack['pen+lbl--top']
        for epoch in range(numIterations):
            pv, v = self.propagateToFinalRBM(inData)
            v = tf.concat([v, labels], axis=1)

            topRBM.cd1(v, numEpochs=1)
            # self.recognize(inData[0:1000], labels[0:1000])

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations, numGibbsIterations, testImgs, testLabels):

        """
        Wake-sleep method for learning all the parameters of network.
        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print("\ntraining wake-sleep..")
        self.n_samples = vis_trainset.shape[0]
        rbm0 = self.rbm_stack['vis--hid']
        print(rbm0.weight_v_to_h)
        rbm1 = self.rbm_stack['hid--pen']
        rbm2 = self.rbm_stack['pen+lbl--top']
        trainingSetParts = rbm0.divideIntoParts(vis_trainset, 500)
        labelParts = rbm0.divideIntoParts(lbl_trainset, 500)

        for it in range(n_iterations):
            print("Iteration: ", it)
            rbm0Deltas = [[], []]
            rbm1Deltas = [[], []]
            for partIndex in range(len(trainingSetParts)):
                print(partIndex, "/", len(trainingSetParts))
                trainPart, labelPart = trainingSetParts[partIndex], labelParts[partIndex]

                ph0, h0 = rbm0.get_h_given_v(trainPart)
                ph1, h1 = rbm1.get_h_given_v(h0)

                v2 = tf.identity(h1)
                v2 = tf.concat([v2, labelPart], axis=1)  # Add correct labels
                firstV2 = tf.identity(v2)
                for i in range(numGibbsIterations):  # Should one perform K-gibbs sampling here?
                    ph2, h2 = rbm2.get_h_given_v(v2)
                    pv2, v2 = rbm2.get_v_given_h(h2)
                    if (i == 0):
                        firstH2 = tf.identity(h2)
                    if (i == numGibbsIterations - 2):
                        lastV2 = tf.identity(v2)

                v2 = v2[:, :-10]  # Remove the label nodes afterwards

                pv1, v1 = rbm1.get_v_given_h(v2)
                pv0, v0 = rbm0.get_v_given_h(v1)

                # Wake training
                pv0T, v0T = rbm0.get_v_given_h(h0)
                rbm0.update_generate_params(h0, trainPart, pv0T)
                # rbm0Deltas[0].append(rbm0.calcDelta(h0, trainPart, pv0T))
                pv1T, v1T = rbm1.get_v_given_h(h1)
                rbm1.update_generate_params(h1, h0, pv1T)
                # rbm1Deltas[0].append(rbm1.calcDelta(h1, h0, pv1T))

                # Sleep training
                ph1T, h1T = rbm1.get_h_given_v(v1)
                rbm1.update_recognize_params(v1, v2, ph1T)
                # rbm1Deltas[1].append(rbm1.calcDelta(v1, v2, ph1T))
                ph0T, h0T = rbm0.get_h_given_v(v0)
                rbm0.update_recognize_params(v0, v1, ph0T)
                # rbm0Deltas[1].append(rbm0.calcDelta(v0, v1, ph0T))

                # print(firstV2.shape)
                # print(firstH2.shape)
                # print(lastV2.shape)
                # print(ph2.shape)
                # print(rbm2.delta_bias_v)
                # print(rbm2.delta_bias_h)
                rbm2.update_params(firstV2, firstH2, lastV2, ph2)

            '''
            rbm0.applyRecognizeDeltas(rbm0Deltas[1])
            rbm0.applyGenerativeDeltas(rbm0Deltas[0])

            rbm1.applyRecognizeDeltas(rbm1Deltas[1])
            rbm1.applyGenerativeDeltas(rbm1Deltas[0])
            '''

            self.recognize(testImgs, testLabels)
            # [TODO TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.

            # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.

            # [TODO TASK 4.3] sleep phase : from the activities in the top RBM, drive the network top to bottom.

            # [TODO TASK 4.3] compute predictions : compute generative predictions from wake-phase activations, and recognize predictions from sleep-phase activations.
            # Note that these predictions will not alter the network activations, we use them only to learn the directed connections.

            # [TODO TASK 4.3] update generative parameters : here you will only use 'update_generate_params' method from rbm class.

            # [TODO TASK 4.3] update parameters of top rbm : here you will only use 'update_params' method from rbm class.

            # [TODO TASK 4.3] update generative parameters : here you will only use 'update_recognize_params' method from rbm class.

        print("Loop finished")

    def loadStackWeights(self, trainingStep):
        # if (trainingStep <= 0):
        #    print("Pre-loading nothing")
        #    return
        baseDir = "Labbabbab4/AddeJoppeFrallan/"

        '''
        self.rbm_stack['vis--hid'].fixDeltaWeights()
        self.rbm_stack['hid--pen'].fixDeltaWeights()
        self.rbm_stack['pen+lbl--top'].fixDeltaWeights()
        '''

        if (trainingStep > 0):
            print("Loading RBN-Layer-0 Weights")
            self.rbm_stack['vis--hid'].load_weights(baseDir + "DBN-RBM-0-Weights")
            print(self.rbm_stack['vis--hid'].weight_v_to_h)
        if (trainingStep > 1):
            print("Loading RBN-Layer-1 Weights")
            self.rbm_stack['hid--pen'].load_weights(baseDir + "DBN-RBM-1-Weights")
        if (trainingStep > 2):
            print("Loading RBN-Layer-2 Weights")
            self.rbm_stack['pen+lbl--top'].load_weights(baseDir + "DBN-RBM-2-Weights")

        '''
        self.rbm_stack['vis--hid'].fixDeltaWeights()
        self.rbm_stack['hid--pen'].fixDeltaWeights()
        self.rbm_stack['pen+lbl--top'].fixDeltaWeights()
        # Fixing previous bug
        '''

    def saveAllWeights(self, baseFileName):
        keys = ['vis--hid', 'hid--pen', 'pen+lbl--top']
        for i, k in enumerate(keys):
            self.rbm_stack[k].save_weights(baseFileName.format(i))

    def loadAllWeights(self, baseFileName):
        keys = ['vis--hid', 'hid--pen', 'pen+lbl--top']
        for i, k in enumerate(keys):
            self.rbm_stack[k].load_weights(baseFileName.format(i))
