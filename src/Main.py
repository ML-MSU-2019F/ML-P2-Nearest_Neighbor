from DataSet import DataSet
from NearestNeigbor.PAM import PAM
from NearestNeigbor.CondensedNearestNeighbor import CondensedNearestNeighbor
from NearestNeigbor.KMeans import KMeans
from MultiLayerPerceptron import MultiLayerPerceptron
from Backprop import Backprop
from GeneticAlgorithm import GeneticAlgorithm

# Main in this project is used to stage and run algorithms.  The algorithms for this project that are being run are:
# CNN, KMeans, PAM - MLP, and RBF.
# All the examples will be uncommented, it is highly recommended you comment out all datasets and algorithms except
# the ones you want
def main():
    backprop = Backprop()
    genetic_algorithm = GeneticAlgorithm()
    # ========Classification
    abalone = DataSet("../data/abalone.data", 0, regression=False)
    ffn = MultiLayerPerceptron(8, 0, [], 3, learning_rate=.00001, momentum_constant=.4, stop_accuracy=.0001)
    ffn.setLearningAlgorithm(genetic_algorithm)
    abalone.runAlgorithm(ffn)

    cars = DataSet("../data/car.data", target_location=6, isCars=True, regression=False);
    ffn = MultiLayerPerceptron(6, 0, [], 4, learning_rate=.00001, momentum_constant=.1, stop_accuracy=.0001)
    cars.runAlgorithm(ffn)

    segmentation = DataSet("../data/segmentation.data", target_location=0, regression=False)
    # cnn, means, medoids = getCNNMeansAndMedoids(segmentation, 5)
    # runRadialBasisNetworkAlgorithms(cnn, means, medoids, segmentation, 19, 7, .001)
    # print("Initializing Radial Basis Network with CNN Set")
    ffn = MultiLayerPerceptron(19, 2, [8, 8], 7, learning_rate=.01, momentum_constant=.8, stop_accuracy=.0001)
    segmentation.runAlgorithm(ffn)

    # ===========Regression
    print("=====Forest Fire Regression (Area Burned (hectares))=====")
    forest_fire = DataSet("../data/forestfires.data", target_location=12, dates=2, days=3, regression=True)
    # # best for zero layer forest fire: MultiLayerPerceptron(12, 0, [], 1, learning_rate=.00001, momentum_constant=.2)
    # # best for 1 layer MultiLayerPerceptron(12, 1, [12], 1, learning_rate=.00001, momentum_constant=.2)
    ffn = MultiLayerPerceptron(12, 2, [6, 6], 1, learning_rate=0.001, momentum_constant=.6, stop_accuracy=.1)
    forest_fire.runAlgorithm(ffn)

    print("=====Machine Performance Regression (relative performance)=====")
    machine = DataSet("../data/machine.data", target_location=7, ignore=[0, 1], regression=True)
    # knn = KNearestNeighbor(5)
    # machine.runAlgorithm(knn)
    ffn = MultiLayerPerceptron(7, 2, [8, 8], 1, learning_rate=.001, momentum_constant=.2)
    machine.runAlgorithm(ffn)

    print("=====Wine Quality=====")
    wine = DataSet("../data/winequality.data", target_location=11, regression=True)
    # wine zero best: MultiLayerPerceptron(11, 0, [], 1, learning_rate=.00001, momentum_constant=.2)
    ffn = MultiLayerPerceptron(11, 2, [5, 5], 1, learning_rate=0.0001, momentum_constant=.1)
    wine.runAlgorithm(ffn)


def runRadialBasisNetworkAlgorithms(cnn, kmeans, medoids, data_set, inputs, outputs,learning_rate):
    print("Running CNN Initialized Radial Basis Network")
    cnn_rbf = RadialBasisNetwork(data_set, cnn, inputs, outputs, learning_rate)
    print("Running K-Means Initialized Radial Basis Network")
    kmeans_rbf = RadialBasisNetwork(data_set, kmeans, inputs, outputs, learning_rate)
    print("Running PAM Initialized Radial Basis Network")
    medoids_rbf = RadialBasisNetwork(data_set, medoids, inputs, outputs, learning_rate)

def getCNNMeansAndMedoids(data_set, k):
    cnn = CondensedNearestNeighbor(k)
    data_set.runAlgorithm(cnn)
    cnn_set = data_set.algo_result
    means = KMeans(data_set).centroids
    medoids = PAM(data_set).medoids
    return cnn_set, means, medoids

if __name__ == '__main__':
    main()