from LearningAlgorithm import LearningAlgorithm
from MultiLayerPerceptron import MultiLayerPerceptron
import copy
import random
import math
import heapq
import numpy
import time

class GeneticAlgorithm(LearningAlgorithm):
    def __init__(self, population_count=10, mutation_rate=.2, mutation_shift_constant=.3, crossover_rate=.5):
        self.population_count = population_count
        self.population = None
        self.data_set = None
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_shift_constant = mutation_shift_constant
        self.weight_length = None

    def run(self, mlp: MultiLayerPerceptron):
        self.data_set = mlp.data_set
        self.weight_length = len(mlp.weights)
        self.data_set.makeRandomMap(self.data_set.data, 5)
        losses = []
        times = []
        for i in range(0, 5):
            train_set = self.data_set.getRandomMap(i)
            test_set = self.data_set.getAllRandomExcept(i)
            start = time.time()
            print("Running Fold {}".format(i))
            print("=== Initializing Population ===")
            population = self.initPopulation(mlp)
            print("=== Population Initialized ===")
            print("=== Starting GA=== ")
            loss = self.runGA(population, train_set,test_set)
            end = time.time()
            times.append(end-start)
            losses.append(loss)
        mean_loss = numpy.mean(losses)
        mean_time = numpy.mean(times)
        print("Mean MSE: {}".format(mean_loss))
        print("Mean Time: {}".format(mean_time))
        self.weight_length = len(mlp.weights)

    def runGA(self, population, train_set, test_set):
        accuracies = []
        for i in range(0, len(population)):
            accuracy = self.evaluateFitness(population[i], train_set)
            accuracies.append((accuracy, i))
        gen_counter = 0
        stagnation_counter = 0
        best_accuracy_test = math.inf
        best_accuracy_test_mlp = None
        worse_accuracy_counter = 0
        print("Gen: ", end="")
        while True:
            print("{} ".format(gen_counter), end="")
            gen_counter += 1
            child = self.crossover(accuracies, population)
            population.append(child)
            population = self.mutate(population)
            best_accuracy_train = math.inf
            best_accuracy_train_mlp = None
            best_changed = False
            accuracies = []
            worst = -math.inf
            worst_index = None
            for i in range(0, len(population)):
                accuracy = self.evaluateFitness(population[i], train_set)
                if accuracy > worst:
                    worst = accuracy
                    worst_index = i
                if accuracy < best_accuracy_train:
                    best_accuracy_train = accuracy
                    best_accuracy_train_mlp = population[i]
                    best_changed = True
                # include index for sorting purposes
                accuracies.append((accuracy,i))
            if not best_changed:
                stagnation_counter += 1
            else:
                stagnation_counter = 0
            if stagnation_counter is 10:
                print("\nFinished: Stagnation occurred over 10 generations")
                break
            if gen_counter == 500:
                print("\nFinished: Generation limit reached")
                break
            accuracy_test = self.evaluateFitness(best_accuracy_train_mlp, test_set)
            if best_accuracy_test < accuracy_test:
                worse_accuracy_counter += 1
            else:
                best_accuracy_test_mlp = copy.deepcopy(best_accuracy_train_mlp)
                worse_accuracy_counter = 0
            if worse_accuracy_counter == 5:
                print("\nFinished: Worse accuracy on test set 5 times in a row")
                break
            if accuracy_test < best_accuracy_test:
                best_accuracy_test = accuracy_test
                print("\nNew best accuracy: {}".format(best_accuracy_test))
                print("Gen: ", end="")
            accuracies = self.removeLargestAccuracy(accuracies, worst_index)
            del population[worst_index]
        total = self.evaluateFitness(best_accuracy_test_mlp, test_set)
        print("Final total accuracy: {}".format(total))
        return total

    def removeLargestAccuracy(self, accuracy, index):
        for i in range(0, len(accuracy)):
            if accuracy[i][1] > index:
                accuracy[i] = (accuracy[i][0], accuracy[i][1]-1)
        del accuracy[index]
        return accuracy

    def crossover(self, accuracy, population):
        best = self.tournamentSelect(accuracy)
        pop1 = population[best[0][1]]
        pop2 = population[best[1][1]]
        child = copy.deepcopy(pop1)
        for i in range(0, self.weight_length):
            crossover_rand = random.random()
            if crossover_rand <= self.crossover_rate:
                pop2_weight = pop2.weights[i].weight
                child.weights[i].setWeight(pop2_weight)
        return child

    def tournamentSelect(self, accuracy):
        random_indexes = []
        while len(random_indexes) != math.floor(self.population_count/2):
            rand = random.random()  # rand int between 0.0 and 1.0
            index = math.floor(rand * (len(accuracy) - 1))
            accepted = True
            for j in range(0, len(random_indexes)):
                if random_indexes[j] == index:
                    accepted = False
            if accepted:
                random_indexes.append(index)
        tournament = []
        for i in range(0, len(random_indexes)):
            # since accuracy is a tuple containing the original index, we can use that as a reference
            tournament.append(accuracy[random_indexes[i]])
        heapq.heapify(tournament)
        best = heapq.nsmallest(2, tournament)
        return best

    def mutate(self, population):
        for i in range(0, len(population)):
            for j in range(0, self.weight_length):
                mutate_rand = random.random()
                # mutate if mutation rate is less than or equal to the rand
                if mutate_rand <= self.mutation_rate:
                    rand1 = random.random()  # rand int between 0.0 and 1.0
                    rand2 = random.random()  # rand int between 0.0 and 1.0
                    population_index = math.floor(rand1 * (len(population)-1))
                    weight_index = math.floor(rand2 * (self.weight_length - 1))
                    original_weight = population[i].weights[j].weight
                    update = population[population_index].weights[weight_index].weight * self.mutation_shift_constant
                    # set mutate by getting a random weight from the population and shifting the weight over to it
                    # based on the mutation_shift_constant
                    population[i].weights[j].setWeight(original_weight + update)
        return population

    def evaluateFitness(self, mlp, set):
        accuracy = mlp.checkAccuracyAgainstSet(set, mlp.regression)
        return accuracy

    def initPopulation(self, mlp):
        population = []
        for i in range(0, self.population_count):
            mlp_copy = copy.deepcopy(mlp)
            mlp_copy.initializeWeights()
            mlp_copy.getWeightReferences()
            population.append(mlp_copy)
        return population