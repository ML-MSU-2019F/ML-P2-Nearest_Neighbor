from abc import ABC, abstractmethod
import copy


class LearningAlgorithm(ABC):

    def run(self, mlp):
        pass

    def makeCopies(self, mlp, copy_count):
        copies = []
        for i in range(0, copy_count):
            mlp_copy = copy.deepcopy(mlp)
            mlp_copy.initializeWeights()
            mlp_copy.getWeightReferences()
            copies.append(mlp_copy)
        return copies

    def flatten_weight(self, weights):
        weight_array = []
        for i in range(0, len(weights)):
            weight_array.append(weights[i].weight)
        return weight_array

    def evaluateFitness(self, mlp, set):
        accuracy = mlp.checkAccuracyAgainstSet(set, mlp.regression)
        return accuracy
