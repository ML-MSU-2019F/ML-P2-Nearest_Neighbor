from LearningAlgorithm import LearningAlgorithm
from MultiLayerPerceptron import MultiLayerPerceptron
import copy
import random
import math
import numpy

class DifferentialEvolution(LearningAlgorithm):

    def __init__(self, population=10, replacement_p=.2, beta=.2):
        self.population_count = population
        self.replacement_p = replacement_p
        self.beta = beta
        self.data_set = None
        self.test_set = None
        self.population = None

    def run(self, mlp: MultiLayerPerceptron):
        self.data_set = mlp.data_set
        self.population = self.initPopulation(mlp)
        self.data_set.makeRandomMap(self.data_set.data, 10)
        self.test_set = self.data_set.getRandomMap(0)
        current_population = self.population
        next_population = []
        bound = (len(self.population) -1)
        gen_count = 0
        best_accuracy = math.inf
        while gen_count != 100:
            print("Generation: {}".format(gen_count))
            gen_count+=1
            for i in range(0, len(current_population)):
                initial_accuracy = self.evaluateFitness(current_population[i])
                orig_vector = self.flatten_weight(current_population[i].weights)
                index1 = math.floor(random.random() * bound)
                index2 = math.floor(random.random() * bound)
                index3 = math.floor(random.random() * bound)
                vector1 = self.flatten_weight(current_population[index1].weights)
                vector2 = self.flatten_weight(current_population[index2].weights)
                vector3 = self.flatten_weight(current_population[index3].weights)
                t_vector = numpy.add(vector1, numpy.multiply(numpy.subtract(vector2, vector3),self.beta))
                gen_copy = copy.deepcopy(current_population[0])
                for j in range(0, len(gen_copy.weights)):
                    replacement_rand = random.random()
                    if replacement_rand < self.replacement_p:
                        gen_copy.weights[j].setWeight(t_vector[j])
                    else:
                        gen_copy.weights[j].setWeight(orig_vector[j])
                final_accuracy = self.evaluateFitness(gen_copy)
                if final_accuracy < initial_accuracy:
                    next_population.append(gen_copy)
                    if final_accuracy < best_accuracy:
                        best_accuracy = final_accuracy
                else:
                    next_population.append(current_population[i])
            print("Generation complete, best accuracy: {}".format(best_accuracy))
            current_population = next_population
            next_population = []

    def flatten_weight(self, weights):
        weight_array = []
        for i in range(0, len(weights)):
            weight_array.append(weights[i].weight)
        return weight_array

    def evaluateFitness(self, mlp: MultiLayerPerceptron):
        accuracy = mlp.checkAccuracyAgainstSet(self.test_set, mlp.regression)
        return accuracy

    def initPopulation(self, mlp):
        population = []
        for i in range(0, self.population_count):
            mlp_copy = copy.deepcopy(mlp)
            mlp_copy.initializeWeights()
            mlp_copy.getWeightReferences()
            population.append(mlp_copy)
        return population
