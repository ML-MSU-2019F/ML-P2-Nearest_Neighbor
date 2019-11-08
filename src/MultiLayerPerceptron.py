from Algorithm import Algorithm
from MLPNode import Node
from Layer import Layer
from DataSet import DataSet
import math
import numpy
import pandas
import time
import matplotlib.pyplot as plt
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
        self.inputs = inputs
        self.hidden_layers = hidden_layers
        self.nodes_by_layers = nodes_by_layers
        self.outputs = outputs
        self.stop_accuracy = stop_accuracy
        self.learning_rate = learning_rate
        self.momentum_constant = momentum_constant
        self.regression = False
        # construct our network, input, hidden, and output layers are made, linked, and then weights are initialized
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
        self.regression = regression
        # make a random map with 5 splits for 5 fold validation
        data_set.makeRandomMap(data_set.data, 5)
        # keep track of total accuracies, when the epoch converged, and the time it takes to run training
        total_accuracies = []
        epoch_converged = []
        time_took = []
        # run through 5 fold validation
        for i in range(0, 5):
            print("Starting fold {}".format(i))
            # set data to the 1/5 split
            test_set = data_set.getRandomMap(i)
            # set data to the 4/5 split
            data = data_set.getAllRandomExcept(i)
            # local epoch errors and testset errors that change with every run
            epoch_error = []
            test_set_errors = []
            # get the headless data, for training
            headless_data = data_set.separateClassFromData(data=data)
            # throw error if we were expecting more or fewer inputs
            if len(headless_data[0]) != self.inputs:
                print("Error, we need to have as many inputs as features")
                print("We have {} features, but {} layers".format(len(headless_data[0]), self.inputs))
                exit(1)
            epoch_counter = 0
            # keeping track of error increases
            increased = 0
            # start timer
            start = time.time()
            # training
            while True:
                epoch_counter += 1
                print("Epoch: {}".format(epoch_counter))
                for index in range(0, len(headless_data)):
                    line = headless_data[index]
                    # input values into the input layer
                    self.runForward(line)
                    # if classification
                    if not self.regression:
                        results, max_index = self.getResults()
                        actual_class = data[index][self.data_set.target_location]
                        distance = self.getDistanceFromWantedResult(results, actual_class)
                        # square it for MSE/Cross Entropy error
                        squared_distance = 0.5 * numpy.power(distance, 2)
                        epoch_error.append(numpy.sum(squared_distance)/len(squared_distance))
                        # run backprop using the -distance as error
                        self.runBackprop(-distance)
                    else:
                        # if regression, get single output, and compare it to the actual value
                        output = self.layers[len(self.layers) - 1].nodes[0].output
                        actual_value = data[index][self.data_set.target_location]
                        distance = actual_value - output
                        # square it for MSE
                        squared_distance = numpy.power(distance, 2)
                        epoch_error.append(numpy.sum(squared_distance))
                        # run - distance through backprop
                        self.runBackprop([-distance])
                latest_accuracy = self.checkAccuracyAgainstSet(test_set, self.regression)
                # print the latest accuracy, this is nice for tracking
                print(latest_accuracy)
                error_length = len(test_set_errors)
                # need to have at least 1 error to start comparing
                if len(test_set_errors) > 0:
                    # If test set error increases, or decreases too little
                    if test_set_errors[error_length - 1] - self.stop_accuracy <= latest_accuracy:
                        # add 1 to our increased counter (to combat oscillations)
                        increased += 1
                        # error increases twice in a row
                        if increased == 2:
                            print("Test set error converged")
                            epoch_converged.append(epoch_counter)
                            break
                    else:
                        # reset error increase counter, we did not increase error
                        increased = 0
                        # add accuracy to the list of accuracies
                        test_set_errors.append(latest_accuracy)
                else:
                    # add accuracy to the list of accuracies
                    test_set_errors.append(latest_accuracy)
            # calculate time elapsed and store it
            end = time.time()
            time_took.append(end-start)
            # get final accuracy and append it to test set
            total_accuracies.append(test_set_errors[len(test_set_errors)-1])
            # reset the network for the next run
            self.resetNetwork()
        # after finishing, print out the average values during 5-fold validation
        print("Average Time: {}".format(numpy.mean(time_took)))
        print("Average Error: {}".format(numpy.mean(total_accuracies)))
        print("Average Epoch End: {}".format(numpy.mean(epoch_converged)))
        # print out and see the accuracies and epoch convergence changes.
        ts = pandas.Series(total_accuracies)
        ts.plot()
        plt.show()
        tse = pandas.Series(epoch_converged)
        tse_plot = tse.plot(title="Forest Fire Training, 2 Hidden Layers")
        tse_plot.set(xlabel='Epoch', ylabel='MSE Accuracy')
        plt.show()

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

    # backprop through all nodes in the network, setting the distance/error first
    def runBackprop(self, distance):
        # manually set output in output layer to the derived MSE distance
        for i in range(0, len(self.layers[len(self.layers) - 1].nodes)):
            self.layers[len(self.layers) - 1].nodes[i].error = distance[i]
        # iterate backprop through all but the output layer
        for i in range(len(self.layers) - 2, -1, -1):
            for j in range(0, len(self.layers[i].nodes)):
                self.layers[i].nodes[j].backprop()

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
    """
    Construct the input layer, including a bias node
    """
    def constructInputLayer(self):
        layer = Layer(len(self.layers))
        for i in range(0, self.inputs):
            # initialize node with index, learning_rate, and momentum constant
            node_i = Node(index=i, learning_rate=self.learning_rate, momentum_constant=self.momentum_constant)
            # set references so node is easily able to access global, and semiglobal variables.
            node_i.setTopNetwork(self)
            node_i.setLayer(layer)
            layer.addNode(node_i)
        # add bias node, which has its output fixed to 1
        node_bias = Node(index=self.inputs, learning_rate=self.learning_rate, momentum_constant=self.momentum_constant)
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
            node_i = Node(index=i, learning_rate=self.learning_rate, momentum_constant=self.momentum_constant)
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
                node_j = Node(index=j, learning_rate=self.learning_rate, momentum_constant=self.momentum_constant)
                # set references so node is easily able to access global, and semiglobal variables.
                node_j.setTopNetwork(self)
                node_j.setLayer(layer)
                layer.addNode(node_j)
            # add bias node, which has its output fixed to 1
            node_bias = Node(index=self.nodes_by_layers[i], learning_rate=self.learning_rate,
                             momentum_constant=self.momentum_constant)
            node_bias.overrideOutput(1)
            node_bias.setTopNetwork(self)
            node_bias.setLayer(layer)
            layer.addNode(node_bias)
            self.layers.append(layer)
