from abc import ABC, abstractmethod
import copy


class LearningAlgorithm(ABC):
    """
    LearningAlgorithms ensures that extensions can be run by an MLP for learning it
    It also holds common functions used by learning algorithms.
    """
    # run defined to provide predictable behavior from descendants
    def run(self, mlp):
        pass

    """
    Make deep copies of an mlp for replication purposes
    """
    def makeCopies(self, mlp, copy_count):
        copies = []
        for i in range(0, copy_count):
            mlp_copy = copy.deepcopy(mlp)
            # reinitialize the weights to new randoms
            mlp_copy.initializeWeights()
            # set new weight reference -IMPORTANT- if not done old useless weight
            # references are kept
            mlp_copy.getWeightReferences()
            copies.append(mlp_copy)
        return copies

    """
    Flatten the dynamic weight structure to an array, is used to perform math based on weight
    structures
    """
    def flatten_weight(self, weights):
        weight_array = []
        for i in range(0, len(weights)):
            weight_array.append(weights[i].weight)
        return weight_array

    """
    Helper function that allows a set to be evaluated for fitness.
    """
    def evaluateFitness(self, mlp, set):
        accuracy = mlp.checkAccuracyAgainstSet(set, mlp.regression)
        return accuracy
