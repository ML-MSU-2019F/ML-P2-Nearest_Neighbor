import numpy
import random
import math

class Node:

    def __init__(self, index, bias=None, learning_rate=.2):
        self.bias = bias
        self.index = index
        self.error = None
        self.learning_rate = learning_rate
        self.distance = None
        self.momentum = 0
        self.derived_times_errors = []
        self.weights = []
        self.layer = None
        self.output = None
        self.override_input = None

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
            w_x = numpy.multiply(outputs, weights)
            total = numpy.sum(w_x)
            activated = self.sigmoid(total)
            self.output = activated

    def backprop(self):
        new_weights = []
        derived_times_errors = []
        next_layer = self.layer.next_layer
        # different handling if we dont have to take the sum of error influence
        if next_layer.is_output_layer:
            for i in range(0, len(self.weights)):
                # output is node associated with weights backprop value
                next_layer_node = next_layer.nodes[i]
                # target - out
                next_error = next_layer_node.error
                # output of next node
                next_output = next_layer_node.output
                # derived activation
                derive_activation = self.sigmoidDerived(next_output)
                derived_times_error = next_error * derive_activation

                # error = next_error * derive_activation * next_output
                error = derived_times_error * self.output
                delta = None
                if isinstance(self.momentum, int):
                    delta = (error*self.learning_rate) + self.momentum
                else:
                    delta = (error * self.learning_rate) + (self.momentum[i] * 0.5)
                weight = self.weights[i] - delta
                new_weights.append(weight)
                derived_times_errors.append(derived_times_error)
        else:
            for i in range(0, len(self.weights)):
                next_layer_node = next_layer.nodes[i]
                next_d_and_error = next_layer_node.derived_times_errors
                error_by_weight = numpy.multiply(next_d_and_error, next_layer_node.weights)
                error_sum = numpy.sum(error_by_weight)
                next_output = next_layer_node.output
                derive_activation = self.sigmoidDerived(next_output)
                derived_times_error = error_sum * derive_activation
                derived_times_errors.append(derived_times_error)
                error = derived_times_error * self.output
                delta = None
                if isinstance(self.momentum, int):
                    delta = (error * self.learning_rate) + self.momentum
                else:
                    delta = (error * self.learning_rate) + (self.momentum[i] * 0.5)
                weight = self.weights[i] - delta
                new_weights.append(weight)
        self.derived_times_errors = numpy.array(derived_times_errors)
        # momentum, new weights - last weights
        self.momentum = numpy.subtract(new_weights, self.weights)
        self.weights = numpy.array(new_weights)

    def sigmoid(self, x):
        return 1.0/(1.0 + numpy.exp(-x))

    def sigmoidDerived(self,x):
        return (x * (1-x))