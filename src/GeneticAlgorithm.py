from LearningAlgorithm import LearningAlgorithm
from MultiLayerPerceptron import MultiLayerPerceptron
import copy
import random
import math
import heapq


class GeneticAlgorithm(LearningAlgorithm):
    def __init__(self, population_count=10, mutation_rate=.01, mutation_shift_constant=.5, crossover_rate=.5):
        self.population_count = population_count
        self.population = None
        self.data_set = None
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_shift_constant = mutation_shift_constant
        self.weight_length = None

    def run(self, mlp: MultiLayerPerceptron):
        self.data_set = mlp.data_set
        self.population = self.initPopulation(mlp)
        self.weight_length = len(mlp.weights)
        accuracy = self.evaluateFitness()
        print(accuracy)
        iter = 0
        while iter < 500:
            print("Gen {}".format(iter))
            iter += 1
            self.crossover(accuracy)
            self.mutate()
            accuracy = self.evaluateFitness()
            heapq.heapify(accuracy)
            largest = heapq.nlargest(1, accuracy)
            accuracy = self.removeLargestAccuracy(accuracy)
            del self.population[largest[0][1]]
            print(accuracy)
        print()

    def removeLargestAccuracy(self, accuracy):
        largest_index = accuracy[len(accuracy)-1][1]
        for i in range(0, len(accuracy)):
            if accuracy[i][1] > largest_index:
                accuracy[i] = (accuracy[i][0], accuracy[i][1]-1)
        del accuracy[len(accuracy) - 1]
        return accuracy

    def crossover(self, accuracy):
        best = self.tournamentSelect(accuracy)
        pop1 = self.population[best[0][1]]
        pop2 = self.population[best[1][1]]
        child = copy.deepcopy(pop1)
        for i in range(0, self.weight_length):
            crossover_rand = random.random()
            if crossover_rand <= self.crossover_rate:
                pop2_weight = pop2.weights[i].weight
                child.weights[i].setWeight(pop2_weight)
        self.population.append(child)

    def tournamentSelect(self, accuracy):
        random_indexes = []
        while len(random_indexes) != 5:
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

    def mutate(self):
        for i in range(0, len(self.population)):
            for j in range(0, self.weight_length):
                mutate_rand = random.random()
                # mutate if mutation rate is less than or equal to the rand
                if mutate_rand <= self.mutation_rate:
                    rand1 = random.random()  # rand int between 0.0 and 1.0
                    rand2 = random.random()  # rand int between 0.0 and 1.0
                    population_index = math.floor(rand1 * (len(self.population)-1))
                    weight_index = math.floor(rand2 * (self.weight_length - 1))
                    original_weight = self.population[i].weights[j].weight
                    update = self.population[population_index].weights[weight_index].weight * self.mutation_shift_constant
                    # set mutate by getting a random weight from the population and shifting the weight over to it
                    # based on the mutation_shift_constant
                    self.population[i].weights[j].setWeight(original_weight + update)

    def evaluateFitness(self):
        self.data_set.makeRandomMap(self.data_set.data, 10)
        subset = self.data_set.getRandomMap(0)
        accuracies = []
        for i in range(0, len(self.population)):
            print("Checking accuracy against population, progress: {:.0f}%".format(i*100/(len(self.population)-1)))
            accuracy = self.population[i].checkAccuracyAgainstSet(subset, self.population[i].regression)
            accuracies.append((accuracy, i))
        return accuracies

    def initPopulation(self, mlp):
        population = []
        for i in range(0, self.population_count):
            mlp_copy = copy.deepcopy(mlp)
            mlp_copy.initializeWeights()
            mlp_copy.getWeightReferences()
            population.append(mlp_copy)
        return population