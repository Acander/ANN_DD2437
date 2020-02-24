from Labbabbab4.codeAlaPawel.util import *

from Labbabbab4.codeAlaPawel.rbm import RestrictedBoltzmannMachine

if __name__ == '__main__':
    image_size = [28, 28]
    train_imgs, train_lbls, test_imgs, test_lbls = read_mnist(dim=image_size, n_train=60000, n_test=10000)

    rbm = RestrictedBoltzmannMachine(ndim_visible=image_size[0] * image_size[1],
                                     ndim_hidden=500,
                                     is_bottom=True,
                                     image_size=image_size,
                                     is_top=False,
                                     n_labels=10,
                                     batch_size=20
                                     )

    rbm.cd1(visible_trainset=train_imgs, n_iterations=10000)
