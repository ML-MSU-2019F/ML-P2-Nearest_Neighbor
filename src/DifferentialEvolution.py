from LearningAlgorithm import LearningAlgorithm
from MultiLayerPerceptron import MultiLayerPerceptron
import copy
import random
import math
import numpy
import time


class DifferentialEvolution(LearningAlgorithm):

    def __init__(self, population=10, replacement_p=.6, beta=.4):
        self.population_count = population
        self.replacement_p = replacement_p
        self.beta = beta
        self.data_set = None
        self.test_set = None
        self.population = None

    def run(self, mlp: MultiLayerPerceptron):
        self.data_set = mlp.data_set
        self.data_set.makeRandomMap(self.data_set.data, 5)
        losses = []
        times = []
        for i in range(0, 5):
            train_set = self.data_set.getRandomMap(i)
            test_set = self.data_set.getAllRandomExcept(i)
            start = time.time()
            print("Running Fold {}".format(i))
            print("=== Initializing Population ===")
            population = self.makeCopies(mlp, self.population_count)
            print("=== Population Initialized ===")
            print("=== Starting GA=== ")
            loss = self.runDE(population, train_set,test_set)
            end = time.time()
            times.append(end-start)
            losses.append(loss)
        mean_loss = numpy.mean(losses)
        mean_time = numpy.mean(times)
        print("Mean MSE: {}".format(mean_loss))
        print("Mean Time: {}".format(mean_time))

    def runDE(self, population, train_set, test_set):
        current_population = population
        next_population = []
        bound = (len(population) - 1)
        gen_count = 0
        best_accuracy = math.inf
        best_mlp = None
        best_test_mlp = None
        accuracy_best_test = math.inf
        print("Generation: ")
        while gen_count != 100:
            stagnation_counter = 0
            worse_accuracy_counter = 0
            last_best_accuracy = accuracy_best_test
            print("{},".format(gen_count), end="")
            gen_count += 1
            for i in range(0, len(current_population)):
                initial_accuracy = self.evaluateFitness(current_population[i], train_set)
                orig_vector = self.flatten_weight(current_population[i].weights)
                index1 = math.floor(random.random() * bound)
                index2 = math.floor(random.random() * bound)
                index3 = math.floor(random.random() * bound)
                vector1 = self.flatten_weight(current_population[index1].weights)
                vector2 = self.flatten_weight(current_population[index2].weights)
                vector3 = self.flatten_weight(current_population[index3].weights)
                t_vector = numpy.add(vector1, numpy.multiply(numpy.subtract(vector2, vector3), self.beta))
                gen_copy = copy.deepcopy(current_population[0])
                for j in range(0, len(gen_copy.weights)):
                    replacement_rand = random.random()
                    if replacement_rand < self.replacement_p:
                        gen_copy.weights[j].setWeight(t_vector[j])
                    else:
                        gen_copy.weights[j].setWeight(orig_vector[j])
                final_accuracy = self.evaluateFitness(gen_copy, train_set)
                if final_accuracy < initial_accuracy:
                    next_population.append(gen_copy)
                    if final_accuracy < best_accuracy:
                        stagnation_counter = 0
                        best_accuracy = final_accuracy
                        best_mlp = copy.deepcopy(gen_copy)
                        test_accuracy = self.evaluateFitness(best_mlp, test_set)
                        if test_accuracy < accuracy_best_test:
                            accuracy_best_test = test_accuracy
                            best_test_mlp = best_mlp
                else:
                    next_population.append(current_population[i])
            if last_best_accuracy is accuracy_best_test:
                stagnation_counter += 1
            if stagnation_counter == 5:
                print("Finished, Accuracy didn't improve for 5 generations")
                break
            print("Generation complete, best accuracy: {}".format(accuracy_best_test))
            current_population = next_population
            next_population = []
        return accuracy_best_test




