import numpy
import pandas
import time
import matplotlib.pyplot as plt
from Network.MultiLayerPerceptron import MultiLayerPerceptron
from LearningAlgorithm import LearningAlgorithm


class Backprop(LearningAlgorithm):
    """
    Backprop algorithm
    Description: uses backprop to learn a MLP provide through the run function
    """
    def __init__(self):
        self.mlp = None

    """
    Run method that is called by the top level LearningAlgorithm class from the MultiLayerPerceptron class
    """
    def run(self, mlp: MultiLayerPerceptron):
        self.mlp = mlp
        data_set = mlp.data_set
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
            if len(headless_data[0]) != mlp.inputs:
                print("Error, we need to have as many inputs as features")
                print("We have {} features, but {} layers".format(len(headless_data[0]), mlp.inputs))
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
                    mlp.runForward(line)
                    # if classification
                    if not mlp.regression:
                        results, max_index = mlp.getResults()
                        actual_class = data[index][mlp.data_set.target_location]
                        distance = mlp.getDistanceFromWantedResult(results, actual_class)
                        # square it for MSE/Cross Entropy error
                        squared_distance = 0.5 * numpy.power(distance, 2)
                        epoch_error.append(numpy.sum(squared_distance) / len(squared_distance))
                        # run backprop using the -distance as error
                        self.runBackprop(-distance)
                    else:
                        # if regression, get single output, and compare it to the actual value
                        output = mlp.layers[len(mlp.layers) - 1].nodes[0].output
                        actual_value = data[index][mlp.data_set.target_location]
                        distance = actual_value - output
                        # square it for MSE
                        squared_distance = numpy.power(distance, 2)
                        epoch_error.append(numpy.sum(squared_distance))
                        # run - distance through backprop
                        self.runBackprop([-distance])
                latest_accuracy = mlp.checkAccuracyAgainstSet(test_set, mlp.regression)
                # print the latest accuracy, this is nice for tracking
                print(latest_accuracy)
                error_length = len(test_set_errors)
                # need to have at least 1 error to start comparing
                if len(test_set_errors) > 0:
                    # If test set error increases, or decreases too little
                    if test_set_errors[error_length - 1] - mlp.stop_accuracy <= latest_accuracy:
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
            time_took.append(end - start)
            # get final accuracy and append it to test set
            total_accuracies.append(test_set_errors[len(test_set_errors) - 1])
            # reset the network for the next run
            mlp.resetNetwork()
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

    # backprop through all nodes in the network, setting the distance/error first
    def runBackprop(self, distance):
        layers = self.mlp.layers
        # manually set output in output layer to the derived MSE distance
        for i in range(0, len(layers[len(layers) - 1].nodes)):
            layers[len(layers) - 1].nodes[i].error = distance[i]
        # iterate backprop through all but the output layer
        for i in range(len(layers) - 2, -1, -1):
            for j in range(0, len(layers[i].nodes)):
                layers[i].nodes[j].backprop()