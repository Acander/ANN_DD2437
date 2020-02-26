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

        self.n_gibbs_recog = 15
        self.n_gibbs_gener = 200
        self.n_gibbs_wakesleep = 5

        self.print_period = 2000

        self.loadStackWeights(self.trainingStep)

    def recognize(self, true_img, true_lbl):

        """Recognize/Classify the data into label categories and calculate the accuracy

        Args:
          true_imgs: visible data shaped (number of samples, size of visible layer)
          true_lbl: true labels shaped (number of samples, size of label layer). Used only for calculating accuracy, not driving the net
        """

        n_samples = true_img.shape[0]
        vis = true_img  # visible layer gets the image data
        lbl = np.ones(true_lbl.shape) / 10.  # start the net by telling you know nothing about labels

        # [TODO TASK 4.2] fix the image data in the visible layer and drive the network bottom to top. In the top RBM, run alternating Gibbs sampling \
        # and read out the labels (replace pass below and 'predicted_lbl' to your predicted labels).
        # NOTE : inferring entire train/test set may require too much compute memory (depends on your system). In that case, divide into mini-batches.

        for _ in range(self.n_gibbs_recog):
            pass

        predicted_lbl = np.zeros(true_lbl.shape)
        print("accuracy = %.2f%%" % (100. * np.mean(np.argmax(predicted_lbl, axis=1) == np.argmax(true_lbl, axis=1))))

    def generate(self, true_lbl, name):

        """Generate data from labels

        Args:
          true_lbl: true labels shaped (number of samples, size of label layer)
          name: string used for saving a video of generated visible activations
        """

        n_sample = true_lbl.shape[0]

        records = []
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        ax.set_xticks([]);
        ax.set_yticks([])

        lbl = true_lbl

        # [TODO TASK 4.2] fix the label in the label layer and run alternating Gibbs sampling in the top RBM. From the top RBM, drive the network \ 
        # top to the bottom visible layer (replace 'vis' from random to your generated visible layer).

        for _ in range(self.n_gibbs_gener):
            vis = np.random.rand(n_sample, self.sizes["vis"])

            records.append([ax.imshow(vis.reshape(self.image_size), cmap="bwr", vmin=0, vmax=1, animated=True,
                                      interpolation=None)])

        anim = stitch_video(fig, records).save("%s.generate%d.mp4" % (name, np.argmax(true_lbl)))

    def train_greedylayerwise(self, vis_trainset, lbl_trainset, n_iterations):
        f = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)
        #vis_trainset = f(vis_trainset)
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

        # print("training pen+lbl--top")
        # self.rbm_stack["hid--pen"].untwine_weights()
        """ 
        CD-1 training for pen+lbl--top 
        """

    def train_wakesleep_finetune(self, vis_trainset, lbl_trainset, n_iterations):

        """
        Wake-sleep method for learning all the parameters of network. 
        First tries to load previous saved parameters of the entire network.

        Args:
          vis_trainset: visible data shaped (size of training set, size of visible layer)
          lbl_trainset: label data shaped (size of training set, size of label layer)
          n_iterations: number of iterations of learning (each iteration learns a mini-batch)
        """

        print("\ntraining wake-sleep..")

        try:

            self.loadfromfile_dbn(loc="trained_dbn", name="vis--hid")
            self.loadfromfile_dbn(loc="trained_dbn", name="hid--pen")
            self.loadfromfile_rbm(loc="trained_dbn", name="pen+lbl--top")

        except IOError:

            self.n_samples = vis_trainset.shape[0]

            for it in range(n_iterations):

                # [TODO TASK 4.3] wake-phase : drive the network bottom to top using fixing the visible and label data.

                # [TODO TASK 4.3] alternating Gibbs sampling in the top RBM for k='n_gibbs_wakesleep' steps, also store neccessary information for learning this RBM.

                # [TODO TASK 4.3] sleep phase : from the activities in the top RBM, drive the network top to bottom.

                # [TODO TASK 4.3] compute predictions : compute generative predictions from wake-phase activations, and recognize predictions from sleep-phase activations.
                # Note that these predictions will not alter the network activations, we use them only to learn the directed connections.

                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_generate_params' method from rbm class.

                # [TODO TASK 4.3] update parameters of top rbm : here you will only use 'update_params' method from rbm class.

                # [TODO TASK 4.3] update generative parameters : here you will only use 'update_recognize_params' method from rbm class.

                if it % self.print_period == 0: print("iteration=%7d" % it)

    def loadStackWeights(self, trainingStep):
        if (trainingStep <= 0):
            print("Pre-loading nothing")
            return

        if (trainingStep > 0):
            print("Loading RBN-Layer-0 Weights")
            self.rbm_stack['vis--hid'].load_weights("DBN-RBM-0-Weights")
        if (trainingStep > 1):
            print("Loading RBN-Layer-1 Weights")
            self.rbm_stack['hid--pen'].load_weights("DBN-RBM-1-Weights")
        if (trainingStep > 2):
            print("Loading RBN-Layer-2 Weights")
            self.rbm_stack['pen+lbl--top'].load_weights("DBN-RBM-2-Weights")
