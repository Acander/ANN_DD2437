from Labbabbab4.codeAlaPawel.dbn import DeepBeliefNet
from Labbabbab4.codeAlaPawel.util import *

def main():
    print("Starting Main")
    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)
    # train_imgs = train_imgs[:1000]
    # train_lbls = train_lbls[:1000]
    dbn = DeepBeliefNet(3, {'vis': 784, 'hid': 500, 'pen': 500, 'top': 2000, 'lbl': 10})
    # dbn.train_greedylayerwise(train_imgs, train_lbls, 20)
    dbn.recognize(test_imgs[0:10000], test_lbls[0:10000])

    # dbn.generate(test_lbls[0:1], "lol")
    # dbn.loadAllWeights("FineTunedWeights-RBN-{}")
    # dbn.train_wakesleep_finetune(train_imgs, train_lbls, 10, 20, test_imgs, test_lbls)
    # dbn.saveAllWeights("FineTunedWeights-RBN-{}")
    # dbn.generate(test_lbls[0:1], "lol2")


if __name__ == '__main__':
    main()
