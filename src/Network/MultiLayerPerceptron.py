from Algorithm import Algorithm
from Network.MLPNode import MLPNode
from Network.Layer import Layer
from DataSet import DataSet
import math
import numpy
import matplotlib.pyplot as plt
from LearningAlgorithm import LearningAlgorithm
plt.close('all')


class MultiLayerPerceptron(Algorithm):
    """
        Feed forward network
        Description: Facilitates the feeding forward and backpropigation of a multilayer perceptron
        Arguments:
            inputs- how many input nodes
            hidden_layers - how many hidden layers
            nodes_by_layer - nodes in each layer specified, ex: hidden layers = 2, nodes_by_layer = [3,5]
            outputs: how many output nodes
            learning_rate: the learning rate of the network
            momentum_constant: the momentum rate of the network
            stop_accuracy: the accuracy gain that is considered negligible, and stopping will occur
    """
    def __init__(self, inputs: int, hidden_layers: int, nodes_by_layers: list, outputs: int, learning_rate, momentum_constant, stop_accuracy):
        # initializing instance variables
        self.layers = []
        self.weights = []
        self.inputs = inputs
        self.hidden_layers = hidden_layers
        self.nodes_by_layers = nodes_by_layers
        self.outputs = outputs
        self.stop_accuracy = stop_accuracy
        self.learning_rate = learning_rate
        self.momentum_constant = momentum_constant
        self.regression = False
        self.learning_algorithm = None
        # construct our network, input, hidden, and output layers are made, linked, and then weights are initialized
        print("Constructing Neural Net")
        self.constructInputLayer()
        self.constructHiddenLayers()
        self.constructOutputLayer()
        self.link_layers()
        self.initializeWeights()
        self.getWeightReferences()
        print("Neural Net constructed")

    def setLearningAlgorithm(self, la: LearningAlgorithm):
        self.learning_algorithm = la

    def run(self, data_set: DataSet, regression=False):
        self.data_set = data_set
        self.regression = regression
        self.learning_algorithm.run(self)
        return

    """
    Get the distance away from the wanted results, and the actual results
    """
    def getDistanceFromWantedResult(self, results, actual_class):
        wanted_results = []
        # use data_sets ordered_classes variable
        output_layer_index = self.data_set.ordered_classes[actual_class]
        # make matrix of what we want, which will be the wanted class being 1, the unwanted being zero
        # ex, class 2 is what we want and we have 5 classes: [0,0,1,0,0] (this assumes softmax is being used)
        for i in range(0, len(results)):
            if output_layer_index is i:
                wanted_results.append(1)
            else:
                wanted_results.append(0)
        # get the distance of what we wanted from what we got
        distance = numpy.subtract(wanted_results, results)
        return distance

    def checkAccuracyAgainstSet(self, test_set, regression):
        # strip class/target off so tests can be done
        headless = self.data_set.separateClassFromData(data=test_set)
        if not regression:
            accuracy = 0
            # iterate through headless data
            for index in range(0, len(headless)):
                # run line through net
                self.runForward(headless[index])
                # measure distance away from wanted results, add error
                results, max_index = self.getResults()
                class_actual = test_set[index][self.data_set.target_location]
                distance = self.getDistanceFromWantedResult(results, class_actual)
                accuracy += numpy.sum(.5*numpy.power(distance, 2))
            return accuracy/len(headless)
        else:
            mse_sum = 0
            # iterate through headless data
            for index in range(0, len(headless)):
                # run line through network
                self.runForward(headless[index])
                # get singular output
                output = self.layers[len(self.layers) - 1].nodes[0].output
                # get what the data says it should be
                actual = test_set[index][self.data_set.target_location]
                # mse of the distance
                mse = numpy.power(actual-output, 2)
                mse_sum += mse
            return mse_sum/len(test_set)

    # run feature through the network
    def runForward(self, line):
        for i in range(0, len(line)):
            # input values into first layer
            self.layers[0].nodes[i].overrideInput(line[i])
        # go though each layer, running based on set input
        for i in range(0, len(self.layers)):
            for j in range(0, len(self.layers[i].nodes)):
                self.layers[i].nodes[j].run()

    # reset network by clearing layers, and reconstructing
    def resetNetwork(self):
        self.layers = []
        self.constructInputLayer()
        self.constructHiddenLayers()
        self.constructOutputLayer()
        self.link_layers()
        self.initializeWeights()

    """
    Get the results of the output layer versus what they should be
    Returns: 
        results - unformatted output layer results
        max_index - the max index of the max value in the results
    """
    def getResults(self):
        layer_length = len(self.layers)
        max_value = -math.inf  # our found max weight
        max_index = None  # our found max index
        results = []  # the results, localized from the final row
        # iterate through nodes in the last row
        for i in range(0, len(self.layers[layer_length - 1].nodes)):
            # get value of single node
            value = self.layers[layer_length - 1].nodes[i].output
            # append value to local result
            results.append(value)
            # if greater than current max, assign to variables index, and new weight
            if value > max_value:
                max_index = i
                max_value = value
        return results, max_index

    """
    Link the layers by setting the prev and next references on the layers according to the array
    """
    def link_layers(self):
        for i in range(0,len(self.layers)):
            # is not last layer
            if i+1 is not len(self.layers):
                self.layers[i].next_layer = self.layers[i+1]
            # is not first layer
            if i-1 is not -1:
                self.layers[i].prev_layer = self.layers[i-1]
    """
    Initialize the weights of the network, default is randomly between -0.3,0.3
    """
    def initializeWeights(self):
        # iterate through all but last layer
        for i in range(0, len(self.layers)):
            # initialize weights on all nodes
            for j in range(0, len(self.layers[i].nodes)):
                self.layers[i].nodes[j].initWeights(-0.3, 0.3)

    def getWeightReferences(self):
        weights = []
        for i in range(0, len(self.layers)):
            for j in range(0, len(self.layers[i].nodes)):
                for k in range(0, len(self.layers[i].nodes[j].weights)):
                    weights.append(self.layers[i].nodes[j].weights[k])
        self.weights = weights

    """
    Construct the input layer, including a bias node
    """
    def constructInputLayer(self):
        layer = Layer(len(self.layers))
        for i in range(0, self.inputs):
            # initialize node with index, learning_rate, and momentum constant
            node_i = MLPNode(index=i, learning_rate=self.learning_rate, momentum_constant=self.momentum_constant)
            # set references so node is easily able to access global, and semiglobal variables.
            node_i.setTopNetwork(self)
            node_i.setLayer(layer)
            layer.addNode(node_i)
        # add bias node, which has its output fixed to 1
        node_bias = MLPNode(index=self.inputs, learning_rate=self.learning_rate, momentum_constant=self.momentum_constant)
        node_bias.setTopNetwork(self)
        node_bias.setLayer(layer)
        node_bias.overrideOutput(1)
        layer.addNode(node_bias)
        layer.setInputLayer(True)
        self.layers.append(layer)
    """
    Construct the output layer
    """
    def constructOutputLayer(self):
        # this layer will have index of the length
        layer = Layer(len(self.layers))
        for i in range(0, self.outputs):
            # initialize node with index, learning rate and momentum constant
            node_i = MLPNode(index=i, learning_rate=self.learning_rate, momentum_constant=self.momentum_constant)
            # set references so node is easily able to access global, and semiglobal variables.
            node_i.setTopNetwork(self)
            node_i.setLayer(layer)
            layer.addNode(node_i)
        # set as output layer
        layer.setOutputLayer(True)
        # append layer to our layer list
        self.layers.append(layer)

    """
        Construct hidden layers based on arguments passed to init.
        Will exit with error if arguments seem incorrect
    """
    def constructHiddenLayers(self):
        # should have as many specified hidden nodes per layer as specified hidden layers
        if len(self.nodes_by_layers) != self.hidden_layers:
            print("Problem occurred: Cannot have unspecified hidden nodes")
            print("\tSpecified nodes in layers: {}".format(len(self.nodes_by_layers)))
            print("\tHidden Layers: {}".format(self.hidden_layers))
            exit(1)
        for i in range(0, self.hidden_layers):
            layer = Layer(len(self.layers))
            for j in range(0, self.nodes_by_layers[i]):
                # initialize node with an index, learning_rate, and momentum constant
                node_j = MLPNode(index=j, learning_rate=self.learning_rate, momentum_constant=self.momentum_constant)
                # set references so node is easily able to access global, and semiglobal variables.
                node_j.setTopNetwork(self)
                node_j.setLayer(layer)
                layer.addNode(node_j)
            # add bias node, which has its output fixed to 1
            node_bias = MLPNode(index=self.nodes_by_layers[i], learning_rate=self.learning_rate,
                                momentum_constant=self.momentum_constant)
            node_bias.overrideOutput(1)
            node_bias.setTopNetwork(self)
            node_bias.setLayer(layer)
            layer.addNode(node_bias)
            self.layers.append(layer)
