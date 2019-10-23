import numpy
import random
import math


class Node:


    def __init__(self, index, bias=None, learning_rate=0.5):
        self.bias = bias
        self.index = index
        self.learning_rate = learning_rate
        self.distance = None
        self.weights = []
        self.layer = None
        self.index = None
        self.output = None
        self.override_input = None
        self.backprop_value = None

    def overrideInput(self, value):
        self.override_input = value

    def setLayer(self, layer):
        self.layer = layer

    def initWeights(self, start_range, end_range):
        if self.layer.is_output_layer:
            return

        if self.layer.next_layer is None:
            print("Error, tried to init output weight without knowing prev layer")
            exit(1)
        weights = []
        for i in range(0, len(self.layer.next_layer.nodes)):
            rand = random.random()  # rand int between 0.0 and 1.0
            total_range = math.fabs(start_range-end_range)  # total range between start and end
            # random weight by adding to start a percentage of whole range
            rand_weight = (rand * total_range) + start_range
            weights.append(rand_weight)
        self.weights = numpy.array(weights)

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
        # if this is the input layer
        if self.override_input is not None:
            # set the output to the activated overridden input
            self.output = self.sigmoid(self.override_input)
        else:
            weights, outputs = self.getPreviousLayerWeightsAndOutputs()
            weights = numpy.array(weights)
            outputs = numpy.array(outputs)
            w_x = numpy.multiply(weights, outputs)
            total = numpy.sum(w_x)
            activated = self.sigmoid(total)
            self.output = activated

    def backprop(self):
        new_weights = []
        for i in range(0, len(self.weights)):
            # get node associated with weight
            output = self.layer.next_layer.nodes[i].backprop_value
            distance = self.layer.next_layer.nodes[i].distance
            score = distance * (output * (1-output))
            delta = score * self.learning_rate
            weight = self.weights[i] - delta
            new_weights.append(weight)
        self.weights = numpy.array(new_weights)

    def sigmoid(self, x):
        return 1.0/(1.0 + numpy.exp(-x))