from LearningAlgorithm import LearningAlgorithm
from MultiLayerPerceptron import MultiLayerPerceptron
import copy


class GeneticAlgorithm(LearningAlgorithm):
    def __init__(self, population_count=20, mutation_rate=.1, mutation_shift_constant=.1):
        self.population_count = population_count
        self.data_set = None
        self.mutation_rate= mutation_rate
        self.mutation_shift_constant = mutation_shift_constant

    def run(self, mlp: MultiLayerPerceptron):
        self.data_set = mlp.data_set
        population = self.initPopulation(mlp)
        self.evaluateFitness(population)
        print()

    def mutate(self, population):
        for i in range(0, len(population)):
            for j in range(0, len(population[i].layers))

    def evaluateFitness(self, population):
        self.data_set.makeRandomMap(self.data_set.data, 10)
        subset = self.data_set.getRandomMap(0)
        accuracies = []
        for i in range(0, len(population)):
            print("Checking accuracy against population, progress: {:.0f}%".format(i*100/(len(population)-1)))
            accuracy = (1 - population[i].checkAccuracyAgainstSet(subset, population[i].regression))
            accuracies.append((accuracy, i))
        return accuracies

    def initPopulation(self, mlp):
        population = []
        for i in range(0, self.population_count):
            mlp_copy = copy.deepcopy(mlp)
            mlp_copy.initializeWeights()
            population.append(mlp_copy)
        return population