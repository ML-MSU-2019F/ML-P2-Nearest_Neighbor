import numpy


class Node():
    def __init__(self, weight=None,bias=None):
        self.weight = weight,
        self.bias = bias,

    # TODO: Rename
    def perceptron(self, x):
        return numpy.dot(x, self.weight) + self.bias

    def sigmoid(self, x):
        return 1.0/(1.0+np.exp(-x))