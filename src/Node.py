import numpy
import random
import math


class Node:
    weights = []
    layer = None,
    index = None,
    output = None,

    def __init__(self, index, bias=None):
        self.bias = bias
        self.index = index

    def setLayer(self, layer):
        self.layer = layer

    def initWeights(self, start_range, end_range):
        if self.layer.next_layer is None:
            print("Error, tried to init output weight without knowing next layer")
            exit(1)
        for i in range(0, len(self.layer.next_layer.nodes)):
            rand = random.random()  # rand int between 0.0 and 1.0
            total_range = math.fabs(start_range-end_range)  # total range between start and end
            # random weight by adding to start a percentage of whole range
            rand_weight = (rand * total_range) + start_range
            self.weights[i] = rand_weight

    # gets previous layer weights and outputs
    def getPreviousLayerWeightsAndOutputs(self):
        weights = []
        outputs = []
        for i in range(0, len(self.layer.prev_layer.nodes)):
            # go to previous nodes, for each node find the weight relating to this node
            weights.append(self.layer.prev_layer.nodes[i].weights[self.index])
            outputs.append(self.layer.prev_layer.nodes[i].output)
        return weights, outputs

    def run(self):
        weights, outputs = self.getPreviousLayerWeightsAndOutputs()
        weights = numpy.array(weights)
        outputs = numpy.array(outputs)
        w_x = numpy.multiply(weights, outputs)
        total = numpy.sum(w_x)
        activated = self.sigmoid(total)
        self.output = activated

    # TODO: Rename
    def perceptron(self, x):
        return numpy.dot(x, self.weight) + self.bias

    def sigmoid(self, x):
        return 1.0/(1.0 + numpy.exp(-x))