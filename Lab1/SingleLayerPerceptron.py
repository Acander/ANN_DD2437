from Lab1 import Utils
import numpy as np
import pylab as pb

""" Note: a common mistake when implementing this is to accidentally 
orient the matrixes wrongly so that columns and rows are interchanged. 
Make a sketch on paper where you write down the sizes of all components 
starting by the input and how the dimensionality propagatesto the weights to the output."""

NUMBER_OF_DATA_POINTS = 100
EPOCHS = 20
LEARNING_RATE_n = 0.001

"""The weights are stored in matrix W with as many columns as the dimensionality of the 
input patterns and with the number of rows matching the number of the outputs 
(dimensionality of the output)."""

NUMBER_OF_DIM = 2
NUMBER_OF_COLUMNS = 1
NUMBER_OF_ROWS = NUMBER_OF_DIM + 1

def main():
    w_matrix = np.matrix[init_w_vectors_normal(0, 1)]
    classA, classB = init_training_data_normal()

    for e in range(EPOCHS):
        w_matrix = epoch_iteration(w_matrix, classA, classB)


def epoch_iteration(w_matrix, classA, classB):
    delta_w = delta_rule(w_matrix, classA, classB)
    w_matrix = w_matrix - delta_w
    return w_matrix

def delta_rule(weights, patterns, targets):
    delta_w = -LEARNING_RATE_n*np.dot((np.dot(weights, patterns) - targets), np.transpose(patterns))
    return delta_w


def init_w_vectors_normal(mean, variance):
    w_matrix = np.random.normal(mean, variance, (NUMBER_OF_ROWS, NUMBER_OF_COLUMNS))
    return w_matrix

#TODO
"Implement training data initialization using normal distribution"
def init_training_data_normal():
    n = NUMBER_OF_DATA_POINTS
    mA = [2, 1]
    sigmaA = 0.5
    mB = [-2, -1]
    sigmaB = 0.5
    classA = np.zeros(3)
    classB = np.zeros(3)

    classA[0] = np.random.normal(mA[0], sigmaA, n)
    classA[1] = np.random.normal(mA[1], sigmaA, n)
    classA[2] = np.zeros(NUMBER_OF_COLUMNS) + 1

    classB[0] = np.random.normal(mB[0], sigmaB, n)
    classB[1] = np.random.normal(mB[1], sigmaB, n)
    classB[2] = np.zeros(NUMBER_OF_COLUMNS) + 1

    return classA, classB

"""
def init_training_data():
    patterns = [[0, 1, 1],
                [-1, 3, 1],
                [2, 2, 1],
                [-2, -4, 1],
                [6, 7, 1],
                [-2, -1, 1],
                [2, 4, 1],
                [0, 3, 1]]
    patterns = make_matrix_compatible(patterns)

    targets = [1, -1, 1, -1, 1, -1, 1, -1]

    return patterns, targets
"""
def init_test_data():
    patterns = [[0, 4], [2, 4], [9, 9], [-1, -2], [-4, -5], [-2, -2]]
    patterns = make_matrix_compatible(patterns)
    return patterns

def make_matrix_compatible(matrix):
    matrix = np.matrix(matrix)
    matrix = matrix.transpose()
    return matrix

def plot_outcome():
    pb.