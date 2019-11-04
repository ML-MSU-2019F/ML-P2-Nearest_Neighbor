from Algorithm import Algorithm
from Node import Node
from Layer import Layer
from DataSet import DataSet
import math
import numpy
import pandas
import multiprocessing
import matplotlib.pyplot as plt
plt.close('all')


class FeedForwardNetwork(Algorithm):

    def __init__(self, inputs: int, hidden_layers: int, nodes_by_layers: list, outputs: int):
        self.layers = []
        self.inputs = inputs
        self.hidden_layers = hidden_layers
        self.nodes_by_layers = nodes_by_layers
        self.outputs = outputs
        print("Constructing Neural Net")
        self.constructInputLayer()
        self.constructHiddenLayers()
        self.constructOutputLayer()
        self.link_layers()
        self.initializeWeights()
        print("Neural Net constructed")

    # set to false for now, but will change later
    def run(self, data_set: DataSet, regression=False):
        self.data_set = data_set
        data_set.makeRandomMap(data_set.data, 10)
        test_set = data_set.getRandomMap(0)
        data = data_set.getAllRandomExcept(0)
        headless_data = data_set.separateClassFromData(data=data)

        iter = 0
        epoch_error = []
        test_set_errors = []
        # training
        while iter != 5:
            iter+=1
        # input values into the input layer
            for index in range(0, len(headless_data)):
                line = headless_data[index]
                if len(line) != self.inputs:
                    print("Error, we need to have as many inputs as features")
                    print("We have {} features, but only {} layers".format(len(line), self.inputs))
                    exit(1)
                self.runForward(line)
                results, max_value, max_index = self.getResults()
                actual_class = data[index][self.data_set.target_location]
                distance = self.getDistanceFromWantedResult(results, actual_class)
                # square it for MSE/Cross Entropy error
                squared_distance = 0.5 * numpy.power(distance, 2)
                epoch_error.append(numpy.sum(squared_distance))
                self.runBackprop(distance)
            test_set_errors.append(self.checkAccuracyAgainstSet(test_set))

        ts = pandas.Series(test_set_errors)
        ts.plot()
        plt.show()

    def getDistanceFromWantedResult(self, results, actual_class):
        wanted_results = []
        output_layer_index = self.data_set.ordered_classes[actual_class]
        # make matrix of what we want, which will be the wanted class being 1, the unwanted being zero
        # ex, class 2 is what we want and we have 5 classes: [0,0,1,0,0] (this assumes softmax is being used)
        for i in range(0, len(results)):
            if output_layer_index is i:
                wanted_results.append(1)
            else:
                wanted_results.append(0)
        # get the distance of what we wanted from what we got
        distance = numpy.subtract(results, wanted_results)
        return distance

    def checkAccuracyAgainstSet(self, test_set):
        accuracy = 0
        headless = self.data_set.separateClassFromData(data=test_set)
        for index in range(0, len(headless)):
            self.runForward(headless[index])
            results, max_value, max_index = self.getResults()
            class_actual = test_set[index][self.data_set.target_location]

            if max_index == self.data_set.ordered_classes[class_actual]:
                accuracy += 1
        return accuracy/len(test_set)

    def runForward(self, line):
        for i in range(0, len(line)):
            # input values into first layer
            self.layers[0].nodes[i].overrideInput(line[i])
        # go though each layer, running based on set input
        for i in range(0, len(self.layers)):
            for j in range(0, len(self.layers[i].nodes)):
                self.layers[i].nodes[j].run()

    def runBackprop(self, distance):
        # manually set output in output layer to the derived MSE distance
        for i in range(0, len(self.layers[len(self.layers) - 1].nodes)):
            self.layers[len(self.layers) - 1].nodes[i].error = distance[i]
        for i in range(len(self.layers) - 2, -1, -1):
            for j in range(len(self.layers[i].nodes) - 1, -1, -1):
                self.layers[i].nodes[j].backprop()

    def runSetOfNodes(self, nodes):
        for node in nodes:
            node.run()

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
        # turn results into soft_max
        exp_results = numpy.exp(results)
        soft_max_sum = numpy.sum(exp_results)
        # results = exp_results/soft_max_sum
        # set last layer results to be the softmax sum
        return results, max_value, max_index

    def link_layers(self):
        for i in range(0,len(self.layers)):
            # is not last layer
            if i+1 is not len(self.layers):
                self.layers[i].next_layer = self.layers[i+1]
            # is not first layer
            if i-1 is not -1:
                self.layers[i].prev_layer = self.layers[i-1]

    def inputData(self, input):
        if len(input) is not self.inputs:
            print("Problem occurred, inputs specified: {}, inputs given: {}".format(self.inputs, len(input)))
            exit(1)

    def initializeWeights(self):
        # iterate through all but last layer
        for i in range(0, len(self.layers)):
            # initialize weights on all nodes
            for j in range(0, len(self.layers[i].nodes)):
                self.layers[i].nodes[j].initWeights(-0.5, 0.5)

    def constructInputLayer(self):
        layer = Layer(len(self.layers))
        for i in range(0, self.inputs):
            node_i = Node(index=i)
            node_i.setLayer(layer)
            layer.addNode(node_i)
        layer.setInputLayer(True)
        self.layers.append(layer)


    def constructOutputLayer(self):
        layer = Layer(len(self.layers))
        for i in range(0, self.outputs):
            node_i = Node(index=i)
            node_i.setLayer(layer)
            layer.addNode(node_i)
        layer.setOutputLayer(True)
        self.layers.append(layer)

    def constructHiddenLayers(self):
        if len(self.nodes_by_layers) != self.hidden_layers:
            print("Problem occurred: Cannot have unspecified hidden nodes")
            print("\tSpecified nodes in layers: {}".format(len(self.nodes_by_layers)))
            print("\tHidden Layers: {}".format(self.hidden_layers))
            exit(1)

        for i in range(0, self.hidden_layers):
            layer = Layer(len(self.layers))
            for j in range(0, self.nodes_by_layers[i]):
                node_j = Node(index=j)
                node_j.setLayer(layer)
                layer.addNode(node_j)
            self.layers.append(layer)
