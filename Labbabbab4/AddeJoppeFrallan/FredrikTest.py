from Labbabbab4.codeAlaPawel.dbn import DeepBeliefNet
from Labbabbab4.codeAlaPawel.util import *

if __name__ == '__main__':
    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    dbn = DeepBeliefNet(3, {'vis': 784, 'hid': 500, 'pen': 500, 'top': 2000, 'lbl': 10})
    # dbn.train_greedylayerwise(train_imgs, train_lbls, 20)
    # dbn.recognize(test_imgs, test_lbls)
    dbn.generate(test_lbls[0:1], "lol")
