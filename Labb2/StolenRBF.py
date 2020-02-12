import numpy as np


class RBFN(object):

    def __init__(self, hidden_shape, sigma=1.0, lr=0.05):
        """ radial basis function network
        # Arguments
            input_shape: dimension of the input data
            e.g. scalar functions have should have input_dimension = 1
            hidden_shape: the number
            hidden_shape: number of hidden radial basis functions,
            also, number of centers.
        """
        self.hidden_shape = hidden_shape
        self.sigma = sigma
        self.centers = None
        self.weights = np.random.normal(0, 0.1, (hidden_shape, 1))
        self.lr = lr

    def _kernel_function(self, center, data_point):
        return np.exp(-self.sigma * np.linalg.norm(center - data_point) ** 2)

    def _calculate_interpolation_matrix(self, X):
        """ Calculates interpolation matrix using a kernel_function
        # Arguments
            X: Training data
        # Input shape
            (num_data_samples, input_shape)
        # Returns
            G: Interpolation matrix
        """
        G = np.zeros((len(X), self.hidden_shape))
        for data_point_arg, data_point in enumerate(X):
            for center_arg, center in enumerate(self.centers):
                G[data_point_arg, center_arg] = self._kernel_function(
                    center, data_point)
        return G

    def _select_centers(self, X):
        random_args = np.random.choice(len(X), self.hidden_shape)
        print("Random Args:", random_args)
        centers = X[random_args]
        print("Centers:", centers)
        return centers

    def fit(self, X, Y):
        """ Fits weights using linear regression
        # Arguments
            X: training samples
            Y: targets
        # Input shape
            X: (num_data_samples, input_shape)
            Y: (num_data_samples, input_shape)
        """
        self.centers = self._select_centers(X)  # Selects centers by putting them on random data points
        G = self._calculate_interpolation_matrix(X)
        print(G, G.shape)
        self.weights = np.dot(np.linalg.pinv(G), Y)

    def fit2(self, X, Y):
        G = self._calculate_interpolation_matrix(X)
        pred = np.dot(G, self.weights)
        loss =  pred -Y
        # print(pred.shape, G.shape, self.weights.shape)
        # print("Loss:", loss.shape)
        wGrad = G * loss
        # print("wGrad:", wGrad.shape)
        wGrad = np.mean(wGrad, axis=0).reshape((self.hidden_shape, 1))
        # print("wGrad:", wGrad.shape)
        self.weights -= wGrad * self.lr

        return np.mean(np.abs(loss))

    def setCentroids(self, X):
        self.centers = self._select_centers(X)  # Selects centers by putting them on random data points

    def predict(self, X):
        """
        # Arguments
            X: test data
        # Input shape
            (num_test_samples, input_shape)
        """
        G = self._calculate_interpolation_matrix(X)
        predictions = np.dot(G, self.weights)
        return predictions


if __name__ == '__main__':
    from Labb2 import DataHandler, RBFFunc, RBFNet

    '''

    numberOfPoints = 5
    points = np.array([[np.pi / (numberOfPoints / 2) * i] for i in range(numberOfPoints)])
    '''
    trainX, trainY, testX, testY = DataHandler.generateData(shuffle=False)

    import numpy as np
    import matplotlib.pyplot as plt

    # fitting RBF-Network with data
    numNodes = 50
    model = RBFN(hidden_shape=numNodes, sigma=1.)
    model.setCentroids(trainX)

    for i in range(10000000):
        model.lr *= 0.9995
        print(model.fit2(trainX, trainY), model.lr)

    predY = model.predict(testX)
    print(np.mean(np.abs(predY - testY)))

    y_pred = model.predict(trainX)
    # plotting 1D interpolation
    plt.plot(trainX, trainY, 'b-', label='real')
    plt.plot(trainX, y_pred, 'r-', label='fit')
    plt.legend(loc='upper right')
    plt.title('Interpolation using a RBFN {} Nodes'.format(numNodes))
    plt.show()
