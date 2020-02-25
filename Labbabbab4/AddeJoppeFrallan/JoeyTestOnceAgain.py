from Labbabbab4.codeAlaPawel.util import *

from Labbabbab4.codeAlaPawel.rbm import RestrictedBoltzmannMachine


def benchmark(model, trainImgs):
    import tensorflow as tf
    import timeit
    t = lambda x: tf.convert_to_tensor(x, dtype=tf.float32)

    model.weight_vh = t(model.weight_vh)
    model.bias_v = t(model.bias_v)
    model.bias_h = t(model.bias_h)
    # t = lambda x: x
    v0 = t(np.random.random((20, 784)))
    vk = t(np.random.random((20, 784)))
    h0 = t(np.random.random((20, 500)))
    hk = t(np.random.random((20, 500)))

    # data = tf.convert_to_tensor(trainImgs[:20])
    # data = trainImgs[:20]

    def f():
        model.update_params(v0, h0, vk, hk)

    print("Get V given H", timeit.timeit(f, number=50))


if __name__ == '__main__':
    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0] * image_size[1],
                                     ndim_hidden=500,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20,
                                     learning_rate=0.1
                                     )

    # benchmark(rbm, train_imgs)
    #rbm.cd1(visible_trainset=train_imgs, testSet=test_imgs, numEpochs=1)
    # print(rbm.variables)
    print(rbm.getWeightsInNumpyByName("bias_v"))
