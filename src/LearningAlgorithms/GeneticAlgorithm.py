from LearningAlgorithm import LearningAlgorithm
from Network.MultiLayerPerceptron import MultiLayerPerceptron
import copy
import random
import math
import heapq
import numpy
import time


class GeneticAlgorithm(LearningAlgorithm):
    """
    Genetic algorithm
    Description: uses the genetic algorithm to train a multilayer perceptron provided through the run function
    Arguments:
        population_count - how many members in the population
        mutation_rate - probability of a given gene to mutate
        mutatuion_shift_constant - how much to shift gene to a random value chosen from the population
        crossover_rate - using uniform crossover, the probability that the most fit individual in tournament
                         selection to swap its gene with the second most fit.
    """
    def __init__(self, population_count=10, mutation_rate=.1, mutation_shift_constant=.5, crossover_rate=.3):
        self.population_count = population_count
        self.population = None
        self.data_set = None
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_shift_constant = mutation_shift_constant
        self.weight_length = None

    def run(self, mlp: MultiLayerPerceptron):
        self.data_set = mlp.data_set
        # set the length of the weights for easy access
        self.weight_length = len(mlp.weights)
        # make random map for 5 fold validation
        self.data_set.makeRandomMap(self.data_set.data, 5)
        # hold stats from each fold in 5 fold
        losses = []
        times = []
        generations = []
        for i in range(0, 5):
            # get a test and train set
            train_set = self.data_set.getRandomMap(i)
            test_set = self.data_set.getAllRandomExcept(i)
            # start timer to measure algorithm performance
            start = time.time()
            print("Running Fold {}".format(i+1))
            print("=== Initializing Population ===")
            # population is made by making copies of the current network and re-randomizing weights
            population = self.makeCopies(mlp, self.population_count)
            print("=== Population Initialized ===")
            print("=== Starting GA=== ")
            # run the GA algorithm, get the returned loss and generations
            loss, gens = self.runGA(population, train_set, test_set)
            # add data to the arrays
            end = time.time()
            generations.append(gens)
            times.append(end-start)
            losses.append(loss)
        # take the mean from the tracked values
        mean_gens = numpy.mean(generations)
        mean_loss = numpy.mean(losses)
        mean_time = numpy.mean(times)
        # print out the means from stats
        print("Mean MSE: {:.2f}".format(mean_loss))
        print("Mean Time: {:.2f}".format(mean_time))
        print("Mean Generations: {:.2f}".format(mean_gens))

    """
    Run GA, considering a population, train set, and test_set
    """
    def runGA(self, population, train_set, test_set):
        # get initial loss of the population
        losses = []
        for i in range(0, len(population)):
            loss = self.evaluateFitness(population[i], train_set)
            losses.append((loss, i))
        # track the best test loss and the relative mlp
        best_loss_test = math.inf
        # track how many generations had worse loss
        worse_loss_counter = 0
        gen_counter = 0
        print("Gen: ", end="")
        while True:
            print("{} ".format(gen_counter), end="")
            # create a child through using the losses of the population and the population
            child = self.crossover(losses, population)
            # append child to the population
            population.append(child)
            # mutate the populations
            population = self.mutate(population)
            # locally track the best losses from training
            best_loss_train = math.inf
            best_loss_train_mlp = None
            # reset losses reference
            losses = []
            # keep track of worse member of population
            worst = -math.inf
            worst_index = None
            # iterate through the population and get the losses
            for i in range(0, len(population)):
                loss = self.evaluateFitness(population[i], train_set)
                # if the loss is greater than the worst
                if loss > worst:
                    # it is the new worst
                    worst = loss
                    worst_index = i
                # if the loss is less than the best
                if loss < best_loss_train:
                    # set the best to the new reference
                    best_loss_train = loss
                    best_loss_train_mlp = population[i]
                # make new losses, include index for sorting purposes
                losses.append((loss, i))
            # test the best training mlp against the test set
            loss_test = self.evaluateFitness(best_loss_train_mlp, test_set)
            # if the loss is worse than the best from test
            if loss_test > best_loss_test:
                # add to the worse counter
                worse_loss_counter += 1
                if worse_loss_counter == 25:
                    print("\nFinished: Worse loss on test set 25 times in a row")
                    break
            else:
                # if the loss is better, set the new relative references and reset the counter
                worse_loss_counter = 0
            # if the best training mlp's test loss is better than our current best loss from testing
            if loss_test < best_loss_test:
                # set the new reference and print the change out
                best_loss_test = loss_test
                print("\nNew best loss: {:.4f}".format(best_loss_test))
                print("Gen: ", end="")
            # this losses is used in the next loop, so we need to remove the largest loss
            losses = self.removeLargestloss(losses, worst_index)
            # delete the worst index from the population
            del population[worst_index]
            gen_counter += 1
        # algorithm finished, print final best loss
        print("Final total loss: {:.4f}".format(best_loss_test))
        return best_loss_test, gen_counter

    """
    Is used to remove the largest references in the loss data structure to be representative of the
    population after an element is removed
    """
    def removeLargestloss(self, loss, index):
        for i in range(0, len(loss)):
            if loss[i][1] > index:
                loss[i] = (loss[i][0], loss[i][1]-1)
        del loss[index]
        return loss

    """
    Uniform crossover using tournament selection to select from the population.
    Returns child
    """
    def crossover(self, loss, population):
        # get best two from tournament selection
        best = self.tournamentSelect(loss)
        # get indexes from best, and get individuals from population
        pop1 = population[best[0][1]]
        pop2 = population[best[1][1]]
        # make deepcopy of parent 1
        child = copy.deepcopy(pop1)
        for i in range(0, self.weight_length):
            # generate rand
            crossover_rand = random.random()
            # random favors crossing over
            if crossover_rand <= self.crossover_rate:
                # crossover with second best from tournament
                pop2_weight = pop2.weights[i].weight
                child.weights[i].setWeight(pop2_weight)
        return child

    """
    Tournament selection decided by selecting population_count/2 random indexes without replacement
    """
    def tournamentSelect(self, loss):
        random_indexes = []
        #generate random indexes
        while len(random_indexes) != math.floor(self.population_count/2):
            rand = random.random()  # rand int between 0.0 and 1.0
            index = math.floor(rand * (len(loss) - 1))
            accepted = True
            for j in range(0, len(random_indexes)):
                if random_indexes[j] == index:
                    accepted = False
            if accepted:
                random_indexes.append(index)
        # set up torunament
        tournament = []
        # append losses to the tournament
        for i in range(0, len(random_indexes)):
            # since loss is a tuple containing the original index, we can use that as a reference
            tournament.append(loss[random_indexes[i]])
        # heapify the tournament
        heapq.heapify(tournament)
        # get the best two from the tournament
        best = heapq.nsmallest(2, tournament)
        return best

    """
    Mutate a population by selecting random genes present within the population and shifting them to be
    closer to the random gene
    """
    def mutate(self, population):
        # for each member of the population
        for i in range(0, len(population)):
            # for each weight from each member
            for j in range(0, self.weight_length):
                mutate_rand = random.random()
                # mutate if mutation rate is less than or equal to the rand
                if mutate_rand <= self.mutation_rate:
                    rand1 = random.random()  # random for selecting a random member of the population
                    rand2 = random.random()  # random for selecting a random weight from that member
                    population_index = math.floor(rand1 * (len(population)-1))
                    weight_index = math.floor(rand2 * (self.weight_length - 1))
                    # get the original weight
                    original_weight = population[i].weights[j].weight
                    # get the weight of the random gene times the mutation shift constant
                    update = population[population_index].weights[weight_index].weight * self.mutation_shift_constant
                    # set the weight to the new gene made by the update + the original weight
                    population[i].weights[j].setWeight(original_weight + update)
        return population
