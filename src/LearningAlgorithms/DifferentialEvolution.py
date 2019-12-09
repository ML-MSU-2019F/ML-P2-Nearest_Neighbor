from LearningAlgorithm import LearningAlgorithm
from Network.MultiLayerPerceptron import MultiLayerPerceptron
import copy
import random
import math
import numpy
import time


class DifferentialEvolution(LearningAlgorithm):
    """
    Differential Evolution algorithm
    Description: uses differential evolution algorithm to train a MLP provided through the run function
    Arguments:
        population count - how many members in the population
        replacement_p - using uniform crossover, what percent chance does the target vector have to crossover
        beta - beta parameter used in differential evolution
    """
    def __init__(self, population_count=10, replacement_p=.9, beta=.8):
        self.population_count = population_count
        self.replacement_p = replacement_p
        self.beta = beta
        self.data_set = None
        self.test_set = None
        self.population = None

    """
    Run method that is called by the top level LearningAlgorithm class from the MultiLayerPerceptron class
    """
    def run(self, mlp: MultiLayerPerceptron):
        self.data_set = mlp.data_set
        # make random map for 5 fold validation
        self.data_set.makeRandomMap(self.data_set.data, 5)
        # hold stats from each fold in 5 fold
        losses = []
        times = []
        gen_counts = []
        for i in range(0, 5):
            # get a test and train set
            train_set = self.data_set.getRandomMap(i)
            test_set = self.data_set.getAllRandomExcept(i)
            # start timer to measure algorithm performance
            start = time.time()
            print("Running Fold {}".format(i))
            print("=== Initializing Population ===")
            # population is made by making copies of the current network and re-randomizing weights
            population = self.makeCopies(mlp, self.population_count)
            print("=== Population Initialized ===")
            print("=== Starting GA=== ")
            # run the DE algorithm, get the returned loss and generations
            loss, gens = self.runDE(population, train_set, test_set)
            # add data to the arrays
            gen_counts.append(gens)
            end = time.time()
            times.append(end-start)
            losses.append(loss)
        # take the mean from the tracked values
        mean_gens = numpy.mean(gen_counts)
        mean_loss = numpy.mean(losses)
        mean_time = numpy.mean(times)
        # print out the means from stats
        print("Mean MSE: {}".format(mean_loss))
        print("Mean Time: {}".format(mean_time))
        print("Mean Generations: {}".format(mean_gens))

    """
    Run DE algorithm, considering a population, train set and test set
    """
    def runDE(self, population, train_set, test_set):
        # set current population and next population, since population is generational
        current_population = population
        next_population = []
        # helper for selecting random members of the population
        bound = (len(population) - 1)
        # keep best train mlp to be used with test set
        best_train_mlp = None
        # count how many generation where improvement does not occur
        stagnation_counter = 0
        # track the best test set loss
        loss_best_test = math.inf
        gen_count = 0
        print("Gen: ", end="")
        while True:
            print("{} ".format(gen_count), end="")
            # local best is used to get the best inter-generational MLP
            local_best = math.inf
            for i in range(0, len(current_population)):
                # get the initial loss to compare to the new offspring loss
                initial_loss = self.evaluateFitness(current_population[i], train_set)
                # get original vector and three random indexes for random members of the population
                orig_vector = self.flatten_weight(current_population[i].weights)
                index1 = math.floor(random.random() * bound)
                index2 = math.floor(random.random() * bound)
                index3 = math.floor(random.random() * bound)
                # assign vectors from the random selections
                vector1 = self.flatten_weight(current_population[index1].weights)
                vector2 = self.flatten_weight(current_population[index2].weights)
                vector3 = self.flatten_weight(current_population[index3].weights)
                # construct target vector
                t_vector = numpy.add(vector1, numpy.multiply(self.beta, numpy.subtract(vector2, vector3)))
                # generic copy, all values will be changed to be representative of offspring of the t
                # and original vector
                gen_copy = copy.deepcopy(current_population[0])
                # iterate through the weights
                for j in range(0, len(gen_copy.weights)):
                    replacement_rand = random.random()
                    # random favors the target vector
                    if replacement_rand < self.replacement_p:
                        # set the individual weight to that of the target vector
                        gen_copy.weights[j].setWeight(t_vector[j])
                    else:
                        # set the individual weight to that of the original vector
                        gen_copy.weights[j].setWeight(orig_vector[j])
                # get final train set error from the newly fabricated child
                final_loss = self.evaluateFitness(gen_copy, train_set)
                # if the final error is less than the original error, add the child to the next population
                if final_loss < initial_loss:
                    next_population.append(gen_copy)
                    # if the final loss is better than our local best
                    if final_loss < local_best:
                        # set the new local best references
                        local_best = final_loss
                        best_train_mlp = copy.deepcopy(gen_copy)
                else:
                    # keep the parent, we know we don't have a new better loss
                    next_population.append(current_population[i])
            # calculate test loss from the best training mlp
            test_loss = self.evaluateFitness(best_train_mlp, test_set)
            # if the test loss improves by at least 0.001
            if test_loss + .001 <= loss_best_test:
                # set new loss
                loss_best_test = test_loss
                # reset stagnation counter
                stagnation_counter = 0
                # print out this information
                print("\nNew best loss: {:4f}".format(loss_best_test))
                print("Gen: ", end="")
            else:
                # our loss didn't improve enough, add 1 to counter
                stagnation_counter += 1
            # our population did not improve for 25 generations, stop algorithm
            if stagnation_counter == 25:
                print("Finished, loss didn't improve for 25 generations")
                break
            # our current population is now our current next population
            current_population = next_population
            # next population is emptied
            next_population = []
            gen_count += 1
        return loss_best_test, gen_count
