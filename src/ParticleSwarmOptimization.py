from LearningAlgorithm import LearningAlgorithm
from MultiLayerPerceptron import MultiLayerPerceptron
import copy
import random
import math
import numpy


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
        self.data_set.makeRandomMap(self.data_set.data,10)
        self.test_set = self.data_set.getRandomMap(0)
        self.particles = self.initParticles(mlp)
        # global best
        gbf = math.inf
        gb = None
        # local bests
        pb = []
        velocities = []
        positions = []
        # initialization
        for i in range(0, self.particle_count):
            velocities.append([0])
            fitness = self.evaluateFitness(self.particles[i])
            pb.append(fitness)
            weights = self.flatten_weight(self.particles[i].weights)
            print(gbf)
            if fitness < gbf:
                print("we doin")
                gb = weights
                gbf = fitness
            positions.append(weights)
        iter_count = 0
        while iter_count < 50:
            print("Iteration: {}".format(iter_count))
            for i in range(0, self.particle_count):
                # update velocity
                current_position = self.flatten_weight(self.particles[i].weights)
                lv = velocities[i][iter_count]
                rs = random.random()
                rp = random.random()
                personal = (self.individual_bias * rp * numpy.subtract(pb[i], current_position))
                group = (self.swarm_bias * rs * numpy.subtract(gb, current_position))
                new_velocity = lv + personal + group
                velocities[i].append(new_velocity)
                new_position = numpy.add(new_velocity, current_position)
                # set position of the swarm
                for j in range(0, len(self.particles[i].weights)):
                    self.particles[i].weights[j].setWeight(new_position[j])
                fitness = self.evaluateFitness(self.particles[i])
                if pb[i] > fitness:
                    pb[i] = fitness
                # update position
            new_gb_index = None
            for i in range(0, len(pb)):
                if pb[i] < gbf:
                    new_gb_index = i
            if new_gb_index is not None:
                gb = self.flatten_weight(self.particles[new_gb_index].weights)
                gbf = pb[new_gb_index]
                print("New Global best: {:.4f}".format(gbf))
            iter_count += 1


    def evaluateFitness(self, mlp: MultiLayerPerceptron):
        accuracy = mlp.checkAccuracyAgainstSet(self.test_set, mlp.regression)
        return accuracy

    def flatten_weight(self, weights):
        weight_array = []
        for i in range(0, len(weights)):
            weight_array.append(weights[i].weight)
        return weight_array

    def initParticles(self, mlp):
        particles = []
        for i in range(0, self.particle_count):
            mlp_copy = copy.deepcopy(mlp)
            mlp_copy.initializeWeights()
            mlp_copy.getWeightReferences()
            particles.append(mlp_copy)
        return particles
