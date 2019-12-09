from LearningAlgorithm import LearningAlgorithm
from Network.MultiLayerPerceptron import MultiLayerPerceptron
import copy
import random
import math
import numpy
import time


class ParticleSwarmOptimization(LearningAlgorithm):
    """
    Particle Swarm Optimization algorithm
    Description: uses particle swarm optimization to train a MLP provided through the run function
    Arguments:
        particle_count - how many particles are wanted
        individual_bias - how much weight should be given to the individual best vector
        swarm_bias - how much weight should be given to the swarms best vector
        momentum_constant - how much momentum should each particle have
    """
    def __init__(self, particle_count=10, individual_bias=.00005, swarm_bias=.0005, momentum_constant=.001):
        self.particle_count = particle_count
        self.individual_bias = individual_bias
        self.swarm_bias = swarm_bias
        self.momentum_constant = momentum_constant
        self.particles = None
        self.data_set = None
        self.test_set = None

    def run(self, mlp: MultiLayerPerceptron):
        self.data_set = mlp.data_set
        # make a random map to use 5 fold validation
        self.data_set.makeRandomMap(self.data_set.data, 5)
        # hold stats from each fold in 5 fold
        times = []
        losses = []
        iterations = []
        for i in range(0, 5):
            # get a test and train set
            train_set = self.data_set.getRandomMap(i)
            test_set = self.data_set.getAllRandomExcept(i)
            # start timer to measure algorithm performance
            start = time.time()
            print("Running Fold {}".format(i+1))
            print("=== Initializing Particles ===")
            # particles are made by making copies of the current network and re-randomizing weights
            particles = self.makeCopies(mlp, self.particle_count)
            print("=== Particles Initialized ===")
            print("=== Starting PSO === ")
            # run the PSO algorithm, get the returned loss and generations
            loss, iters = self.runPSO(particles, train_set, test_set)
            # add data to the arrays
            iterations.append(iters)
            end = time.time()
            times.append(end-start)
            losses.append(loss)
        # take the mean from the tracked values
        mean_loss = numpy.mean(losses)
        mean_time = numpy.mean(times)
        mean_iter = numpy.mean(iterations)
        # print out the means from stats
        print("Mean MSE: {:.2f}".format(mean_loss))
        print("Mean Time: {:.2f}".format(mean_time))
        print("Mean Iterations: {:.2f}".format(mean_iter))

    """
    Run the PSO algorithm, considering a set of particles, a train set and a test set
    """
    def runPSO(self, particles, train_set, test_set):
        # global best train fitness, weights, and mlp
        gb_train_fitness = math.inf
        gb_weights = None
        gb_mlp = None
        # count times loss does not improve
        degraded_loss_count = 0
        # test set global fitness
        gb_test_fitness = math.inf
        # personal particle tracking
        pb = []
        velocities = []
        positions = []
        # initialization, set initial velocity to zero, assess fitness, and find global best
        for i in range(0, self.particle_count):
            velocities.append([0])
            fitness = self.evaluateFitness(particles[i], train_set)
            pb.append(fitness)
            weights = self.flatten_weight(particles[i].weights)
            if fitness < gb_train_fitness:
                gb_weights = weights
                gb_train_fitness = fitness
            positions.append(weights)
        # count iterations and print it out in a friendly way
        iter_count = 0
        print("Iteration: ", end="")
        # loop until we reach a plateau
        while True:
            print("{} ".format(iter_count), end="")
            # iterate through each particle
            for i in range(0, self.particle_count):
                # get the original position weight structure
                current_position = self.flatten_weight(particles[i].weights)
                # get the last velocity
                last_velocity = velocities[i][iter_count]
                # initialize the randoms needed for PSO
                swarm_rand = random.random()
                personal_rand = random.random()
                # personal weight calculation
                personal = (self.individual_bias * personal_rand * numpy.subtract(pb[i], current_position))
                # swarm weight calculation
                group = (self.swarm_bias * swarm_rand * numpy.subtract(gb_weights, current_position))
                # add up all of the new weights
                new_velocity = (last_velocity + personal + group)
                # set reference so that velocity from this iteration can be used in the next
                velocities[i].append(new_velocity)
                new_position = numpy.add(new_velocity, current_position)
                # set the current particles position to be the new_position
                for j in range(0, len(particles[i].weights)):
                    particles[i].weights[j].setWeight(new_position[j])
                # evaluate the fitness according to training set
                fitness = self.evaluateFitness(particles[i], train_set)
                # if the new fitness is better than the current particles best, set the new best
                if pb[i] > fitness:
                    pb[i] = fitness
                # update position
            new_gb_index = None
            # see if any personal bests now are better than the current best
            for i in range(0, len(pb)):
                if pb[i] < gb_train_fitness:
                    new_gb_index = i
            # if we have a new best
            if new_gb_index is not None:
                # update references to the best mlp, weights, and fitness
                gb_mlp = copy.deepcopy(particles[new_gb_index])
                gb_weights = self.flatten_weight(particles[new_gb_index].weights)
                gb_train_fitness = pb[new_gb_index]
                # evaluate the test set loss of the new global best
                test_set_loss = self.evaluateFitness(gb_mlp, test_set)
                # if it is better than our current best
                if test_set_loss < gb_test_fitness:
                    # reset the loss count
                    degraded_loss_count = 0
                    # set the new global best test fitness
                    gb_test_fitness = test_set_loss
                    # print out this information
                    print("\nNew best loss: {:.4f}".format(gb_test_fitness))
                    print("Iteration: ", end="")
                else:
                    degraded_loss_count += 1
            else:
                # didn't improve loss
                degraded_loss_count += 1
                # if loss doesn't improve for 20 generations, count plateau as best and break
            if degraded_loss_count > 20:
                print("\nFinished: loss plateaued for 20 generations")
                break
            iter_count += 1
        return gb_train_fitness, iter_count

