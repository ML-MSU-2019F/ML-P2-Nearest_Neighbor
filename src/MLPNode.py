import numpy
import random
import math


class Node:

    def __init__(self, index, learning_rate=0.1, momentum_constant=.5):
        self.index = index
        self.error = None
        self.learning_rate = learning_rate
        self.momentum_constant = momentum_constant;
        self.distance = None
        self.momentum = None
        self.derived_times_errors = []
        self.weights = []
        self.layer = None
        self.output = None
        self.override_input = None
        self.override_ouput = None
        self.network = None

    def setTopNetwork(self, network):
        self.network = network

    def overrideOutput(self, value):
        self.override_ouput = value

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
        # go to all but the last, bias node
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
        if self.override_ouput is not None:
            self.output = self.override_ouput
        elif self.override_input is not None:
            # set the output to the activated overridden input
            self.output = self.sigmoid(self.override_input)
        else:
            weights, outputs = self.getPreviousLayerWeightsAndOutputs()
            weights = numpy.array(weights)
            outputs = numpy.array(outputs)
            w_x = numpy.multiply(outputs, weights)
            total = numpy.sum(w_x)
            if self.network.regression and self.layer.is_output_layer:
                self.output = total
            else:
                activated = self.sigmoid(total)
                self.output = activated


    def backprop(self):
        new_weights = []
        derived_times_errors = []
        next_layer = self.layer.next_layer
        # different handling if we dont have to take the sum of error influence
        if next_layer.is_output_layer:
            # backprop over every output weight, relative to output layer
            for i in range(0, len(self.weights)):
                # output is node associated with weights backprop value
                next_layer_node = next_layer.nodes[i]
                # target - out
                next_error = next_layer_node.error
                # output of next node
                next_output = next_layer_node.output
                # derived activation
                derive_activation = None
                if self.network.regression:
                    derive_activation = 1
                else:
                    derive_activation = self.sigmoidDerived(next_output)
                derived_times_error = next_error * derive_activation

                # error = next_error * derive_activation * next_output
                error = derived_times_error * self.output
                delta = None
                if self.momentum is None:
                    delta = (error*self.learning_rate)
                else:
                    delta = (error * self.learning_rate) + (self.momentum[i] * self.momentum_constant)
                weight = self.weights[i] - delta
                new_weights.append(weight)
                derived_times_errors.append(derived_times_error)
        else:
            # backprop over internal weight, that does not lead to the output layer
            for i in range(0, len(self.weights)):
                next_layer_node = next_layer.nodes[i]
                # take the derivative times error of the next node
                next_d_and_error = next_layer_node.derived_times_errors
                # times that value by the output weights of the next node
                error_by_weight = numpy.multiply(next_d_and_error, next_layer_node.weights)
                # sum it up, this is our error sum
                error_sum = numpy.sum(error_by_weight)
                # next output
                next_output = next_layer_node.output
                # get derived
                derive_activation = self.sigmoidDerived(next_output)
                # times derived time error sum
                derived_times_error = error_sum * derive_activation
                derived_times_errors.append(derived_times_error)
                # get individual error
                error = derived_times_error * self.output
                delta = None
                # basically if momentum exists
                if self.momentum is None:
                    delta = (error * self.learning_rate)
                else:
                    delta = (error * self.learning_rate) + (self.momentum[i] * self.momentum_constant)
                # update the weights
                weight = self.weights[i] - delta
                new_weights.append(weight)
        self.derived_times_errors = numpy.array(derived_times_errors)
        # momentum, new weights - last weights
        self.momentum = numpy.subtract(self.weights, new_weights)
        self.weights = numpy.array(new_weights)

    def sigmoid(self, x):
        return 1.0/(1.0 + numpy.exp(-x))

    def sigmoidDerived(self,x):
        return (x * (1-x))