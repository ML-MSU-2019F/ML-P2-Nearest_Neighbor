import numpy
import random
import math
from Network.Weight import Weight


class MLPNode:
    """
    Node class, does much of the work required for feedforward and backprop
    Initialization:
        learning_rate - the rate by which this node learns
        momentum_constant - the constant of how much momentum to add to the backprop
    """
    def __init__(self, index, learning_rate=0.1, momentum_constant=.5):
        self.index = index
        self.error = None
        self.learning_rate = learning_rate
        self.momentum_constant = momentum_constant;
        self.momentum = None
        self.derived_times_errors = []
        self.weights = []
        self.layer = None
        self.output = None
        self.override_input = None
        self.override_output = None
        self.network = None

    # set the top network associated with this node
    def setTopNetwork(self, network):
        self.network = network

    # override the output (does not get activated)
    def overrideOutput(self, value):
        self.override_output = value

    # override the input (does get activated)
    def overrideInput(self, value):
        self.override_input = value

    # set the layer reference
    def setLayer(self, layer):
        self.layer = layer

    # initialize weights based on a range
    # weights for a node are the weights going into the next layer
    def initWeights(self, start_range, end_range):
        # output layer has no weights
        if self.layer.is_output_layer:
            return
        # no next layer, error, this shouldn't happen
        if self.layer.next_layer is None:
            print("Error, tried to init output weight without knowing prev layer")
            exit(1)
        weights = []
        for i in range(0, len(self.layer.next_layer.nodes)):
            weight = Weight()
            rand = random.random()  # rand int between 0.0 and 1.0
            total_range = math.fabs(start_range-end_range)  # total range between start and end
            # random weight by adding to start a percentage of whole range
            rand_weight = (rand * total_range) + start_range
            weight.setWeight(rand_weight)
            weights.append(weight)
        self.weights = weights

    # gets previous layer weights and outputs
    def getPreviousLayerWeightsAndOutputs(self):
        weights = []
        outputs = []
        # go to previous nodes, for each node get the weights relating to the node and the outputs
        for i in range(0, len(self.layer.prev_layer.nodes)):
            weights.append(self.layer.prev_layer.nodes[i].weights[self.index].weight)
            outputs.append(self.layer.prev_layer.nodes[i].output)
        return weights, outputs

    """
    run manages the feedforward of the network
    """
    def run(self):
        # if output is overridden, don't take prev nodes/weights into account
        if self.override_output is not None:
            self.output = self.override_output
        elif self.override_input is not None:
            # set the output to the activated overridden input
            self.output = self.sigmoid(self.override_input)
        else:
            # get the weights and outputs of previous layer, multiply them together and sum them
            weights, outputs = self.getPreviousLayerWeightsAndOutputs()
            weights = numpy.array(weights)
            outputs = numpy.array(outputs)
            w_x = numpy.multiply(outputs, weights)
            total = numpy.sum(w_x)
            # if this is regression and we are on the output layer (should be linear) then don't use the activation
            # function
            if self.network.regression and self.layer.is_output_layer:
                self.output = total
            else:
                # any other node, activate and set output to the activated
                activated = self.sigmoid(total)
                self.output = activated

    """
    Backprop, uses gradient descent to all output weights in this node.
    Has two different logical structures, one for weights going to the output layer, and one with weights going to 
    hidden layers
    """
    def backprop(self):
        new_weights = []
        # derived times error is needed in subsequent layers for calculating the sum of error
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
                error = derived_times_error * self.output
                # get delta
                delta = (error * self.learning_rate)
                # if momentum is present, calculate and add to delta
                if self.momentum is not None:
                    delta += self.momentum[i] * self.momentum_constant
                # temp weight is updated
                weight = self.weights[i] - delta
                # added to new weights list
                new_weights.append(weight)
                # save calculation for future layers
                derived_times_errors.append(derived_times_error)
        else:
            # backprop over internal weight, that does not lead to the output layer
            for i in range(0, len(self.weights)):
                next_layer_node = next_layer.nodes[i]
                # get the derivative times error of the next node
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

                # get individual error
                error = derived_times_error * self.output
                delta = (error * self.learning_rate)
                # if momentum is present, calculate and add to delta
                if self.momentum is not None:
                    delta += (self.momentum[i] * self.momentum_constant)
                # update the weights
                weight = self.weights[i] - delta
                # add weights to new list
                new_weights.append(weight)
                # save derived times error for future layers
                derived_times_errors.append(derived_times_error)
        # set node instance variables
        self.derived_times_errors = numpy.array(derived_times_errors)
        # momentum calculation
        self.momentum = numpy.subtract(self.weights, new_weights)
        self.weights = numpy.array(new_weights)

    # sigmoid helper function
    def sigmoid(self, x):
        return 1.0/(1.0 + numpy.exp(-x))

    # sigmoid derived helper function
    def sigmoidDerived(self,x):
        return (x * (1-x))