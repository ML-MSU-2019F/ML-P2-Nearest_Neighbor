from DataSet import DataSet
from NearestNeigbor.PAM import PAM
from NearestNeigbor.CondensedNearestNeighbor import CondensedNearestNeighbor
from NearestNeigbor.KMeans import KMeans
from MultiLayerPerceptron import MultiLayerPerceptron
from Backprop import Backprop
from GeneticAlgorithm import GeneticAlgorithm
from DifferentialEvolution import DifferentialEvolution
from ParticleSwarmOptimization import ParticleSwarmOptimization
# Main in this project is used to stage and run algorithms.  The algorithms for this project that are being run are:
# CNN, KMeans, PAM - MLP, and RBF.
# All the examples will be uncommented, it is highly recommended you comment out all datasets and algorithms except
# the ones you want
def main():
    particle_swarm = ParticleSwarmOptimization()
    backprop = Backprop()
    genetic_algorithm = GeneticAlgorithm()
    differential_evolution = DifferentialEvolution()
    # ========Classification
    # abalone = DataSet("../data/abalone.data", 0, regression=False)
    # ffn = MultiLayerPerceptron(8, 0, [], 3, learning_rate=.00001, momentum_constant=.4, stop_accuracy=.0001)
    # ffn.setLearningAlgorithm(particle_swarm)
    # abalone.runAlgorithm(ffn)

    # cars = DataSet("../data/car.data", target_location=6, isCars=True, regression=False);
    # ffn = MultiLayerPerceptron(6, 0, [], 4, learning_rate=.00001, momentum_constant=.1, stop_accuracy=.0001)
    # cars.runAlgorithm(ffn)
    #
    # segmentation = DataSet("../data/segmentation.data", target_location=0, regression=False)
    # ffn = MultiLayerPerceptron(19, 2, [8, 8], 7, learning_rate=.01, momentum_constant=.8, stop_accuracy=.0001)
    # ffn.setLearningAlgorithm(particle_swarm)
    # segmentation.runAlgorithm(ffn)
    # #
    # ===========Regression
    print("=====Forest Fire Regression (Area Burned (hectares))=====")
    forest_fire = DataSet("../data/forestfires.data", target_location=12, dates=2, days=3, regression=True)
    ffn = MultiLayerPerceptron(12, 2, [6, 6], 1, learning_rate=0.001, momentum_constant=.6, stop_accuracy=.1)
    ffn.setLearningAlgorithm(differential_evolution)
    forest_fire.runAlgorithm(ffn)
    #
    # print("=====Machine Performance Regression (relative performance)=====")
    # machine = DataSet("../data/machine.data", target_location=7, ignore=[0, 1], regression=True)
    # ffn = MultiLayerPerceptron(7, 2, [8, 8], 1, learning_rate=.001, momentum_constant=.2, stop_accuracy=0.0001)
    # ffn.setLearningAlgorithm(differential_evolution)
    # machine.runAlgorithm(ffn)

    # print("=====Wine Quality=====")
    # wine = DataSet("../data/winequality.data", target_location=11, regression=True)
    # # wine zero best: MultiLayerPerceptron(11, 0, [], 1, learning_rate=.00001, momentum_constant=.2)
    # ffn = MultiLayerPerceptron(11, 2, [8,8], 1, learning_rate=.00001, momentum_constant=.2, stop_accuracy=0.0001)
    # ffn.setLearningAlgorithm(differential_evolution)
    # wine.runAlgorithm(ffn)

if __name__ == '__main__':
    main()