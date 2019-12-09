from DataSet import DataSet
from Network.MultiLayerPerceptron import MultiLayerPerceptron
from LearningAlgorithms.Backprop import Backprop
from LearningAlgorithms.GeneticAlgorithm import GeneticAlgorithm
from LearningAlgorithms.DifferentialEvolution import DifferentialEvolution
from LearningAlgorithms.ParticleSwarmOptimization import ParticleSwarmOptimization
import winsound
# Main in this project is used to stage and run algorithms.  The algorithms for this project that are being run are:
# CNN, KMeans, PAM - MLP, and RBF.
# All the examples will be uncommented, it is highly recommended you comment out all datasets and algorithms except
# the ones you want
def main():
    # settings for tone to play when execution is finished
    frequency = 2500  # Set Frequency To 2500 Hertz
    duration = 1000  # Set Duration To 1000 ms == 1 second
    backprop = Backprop()
    particle_swarm = ParticleSwarmOptimization(particle_count=20, individual_bias=0.5, swarm_bias=0.25,
                                               momentum_constant=0.1)
    genetic_algorithm = GeneticAlgorithm(population_count=10, mutation_rate=0.1, mutation_shift_constant=0.5,
                                         crossover_rate=0.3)
    differential_evolution = DifferentialEvolution(population_count=10, replacement_p=0.9, beta=0.8)
    # ========Classification
    # abalone = DataSet("../data/abalone.data", 0, regression=False)
    # ffn = MultiLayerPerceptron(8, 0, [], 3, learning_rate=.00001, momentum_constant=.4, stop_accuracy=.0001)
    # ffn.setLearningAlgorithm(particle_swarm)
    # abalone.runAlgorithm(ffn)

    # cars = DataSet("../data/car.data", target_location=6, isCars=True, regression=False);
    # ffn = MultiLayerPerceptron(6, 2, [5, 5], 4, learning_rate=.00001, momentum_constant=.1, stop_accuracy=.0001)
    # ffn.setLearningAlgorithm(particle_swarm)
    # cars.runAlgorithm(ffn)
    #
    segmentation = DataSet("../data/segmentation.data", target_location=0, regression=False)
    ffn = MultiLayerPerceptron(19, 2, [8, 8], 7, learning_rate=.01, momentum_constant=.8, stop_accuracy=.0001)
    ffn.setLearningAlgorithm(genetic_algorithm)
    segmentation.runAlgorithm(ffn)
    # #
    # ===========Regression
    particle_swarm = ParticleSwarmOptimization(particle_count=10, individual_bias=0.00005, swarm_bias=0.0005,
                                               momentum_constant=0.001)
    # print("=====Forest Fire Regression (Area Burned (hectares))=====")
    # forest_fire = DataSet("../data/forestfires.data", target_location=12, dates=2, days=3, regression=True)
    # ffn = MultiLayerPerceptron(12, 2, [6, 6], 1, learning_rate=0.001, momentum_constant=.6, stop_accuracy=.1)
    # ffn.setLearningAlgorithm(particle_swarm)
    # forest_fire.runAlgorithm(ffn)
    #
    # print("=====Machine Performance Regression (relative performance)=====")
    machine = DataSet("../data/machine.data", target_location=7, ignore=[0, 1], regression=True)
    ffn = MultiLayerPerceptron(7, 2, [6, 6], 1, learning_rate=.001, momentum_constant=.2, stop_accuracy=0.0001)
    ffn.setLearningAlgorithm(particle_swarm)
    machine.runAlgorithm(ffn)
    # print("=====Wine Quality=====")
    # wine = DataSet("../data/winequality.data", target_location=11, regression=True)
    # # wine zero best: MultiLayerPerceptron(11, 0, [], 1, learning_rate=.00001, momentum_constant=.2)
    # ffn = MultiLayerPerceptron(11, 2, [5, 5], 1, learning_rate=.00001, momentum_constant=.2, stop_accuracy=0.0001)
    # ffn.setLearningAlgorithm(particle_swarm)
    # wine.runAlgorithm(ffn)

    winsound.Beep(frequency, duration)

if __name__ == '__main__':
    main()