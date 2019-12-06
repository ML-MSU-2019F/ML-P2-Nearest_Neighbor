from LearningAlgorithm import LearningAlgorithm
from MultiLayerPerceptron import MultiLayerPerceptron
import copy
import random
import math
import numpy
import time


class ParticleSwarmOptimization(LearningAlgorithm):
    def __init__(self, particle_count=20, individual_bias=.5, swarm_bias=.25, momentum_constant=.1):
        self.particle_count = particle_count
        self.individual_bias = individual_bias
        self.swarm_bias = swarm_bias
        self.momentum_constant = momentum_constant
        self.particles = None
        self.data_set = None
        self.test_set = None

    def run(self, mlp: MultiLayerPerceptron):
        self.data_set = mlp.data_set
        self.data_set.makeRandomMap(self.data_set.data,5)
        times = []
        losses = []
        for i in range(0, 5):
            train_set = self.data_set.getRandomMap(i)
            test_set = self.data_set.getAllRandomExcept(i)
            start = time.time()
            print("Running Fold {}".format(i))
            print("=== Initializing Particles ===")
            particles = self.makeCopies(mlp, self.particle_count)
            print("=== Particles Initialized ===")
            print("=== Starting PSO === ")
            loss = self.runPSO(particles, train_set,test_set)
            end = time.time()
            times.append(end-start)
            losses.append(loss)
        mean_loss = numpy.mean(losses)
        mean_time = numpy.mean(times)
        print("Mean MSE: {}".format(mean_loss))
        print("Mean Time: {}".format(mean_time))

    def runPSO(self, particles, train_set, test_set):
        # global best
        gbf = math.inf
        gb = None
        g_best_mlp = None;
        # local bests
        pb = []
        velocities = []
        positions = []
        # initialization
        for i in range(0, self.particle_count):
            velocities.append([0])
            fitness = self.evaluateFitness(particles[i], train_set)
            pb.append(fitness)
            weights = self.flatten_weight(particles[i].weights)
            if fitness < gbf:
                gb = weights
                gbf = fitness
            positions.append(weights)
        iter_count = 0
        while iter_count < 50:
            print("Iteration: {}".format(iter_count))
            for i in range(0, self.particle_count):
                # update velocity
                current_position = self.flatten_weight(particles[i].weights)
                lv = velocities[i][iter_count]
                rs = random.random()
                rp = random.random()
                personal = (self.individual_bias * rp * numpy.subtract(pb[i], current_position))
                group = (self.swarm_bias * rs * numpy.subtract(gb, current_position))
                new_velocity = lv + personal + group
                velocities[i].append(new_velocity)
                new_position = numpy.add(new_velocity, current_position)
                # set position of the swarm
                for j in range(0, len(particles[i].weights)):
                    particles[i].weights[j].setWeight(new_position[j])
                fitness = self.evaluateFitness(particles[i], train_set)
                if pb[i] > fitness:
                    pb[i] = fitness
                # update position
            new_gb_index = None
            for i in range(0, len(pb)):
                if pb[i] < gbf:
                    new_gb_index = i
            if new_gb_index is not None:
                g_best_mlp = copy.deepcopy(particles[new_gb_index])
                test_set_accuracy = self.evaluateFitness(g_best_mlp, test_set)
                print("Test Set accuracy: {}".format(test_set_accuracy))
            if new_gb_index is not None:
                gb = self.flatten_weight(particles[new_gb_index].weights)
                gbf = pb[new_gb_index]
                print("New Global best: {:.4f}".format(gbf))
            iter_count += 1
        return gbf

