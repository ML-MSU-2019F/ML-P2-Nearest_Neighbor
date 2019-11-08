from DataSet import DataSet
from NearestNeigbor.KNearestNeighbor import KNearestNeighbor
from NearestNeigbor.PAM import PAM
from NearestNeigbor.CondensedNearestNeighbor import CondensedNearestNeighbor
from NearestNeigbor.KMeans import KMeans
from RadialBasisNetwork import RadialBasisNetwork
from MultiLayerPerceptron import FeedForwardNetwork
import math
import time
def main():
    # ======Regression:
    # print("=====Wine Quality=====")
    # wine = DataSet("../data/winequality.data", target_location=11, regression=True)
    # runRegressionAlgorithms(wine)
    #
    # print("=====Forest Fire Regression (Area Burned (hectares))=====")
    # forest_fire = DataSet("../data/forestfires.data", target_location=12, dates=2, days=3, regression=True)
    # runRegressionAlgorithms(forest_fire)
    #


    # #========Classification
    # print("=====Abalone Classification of Sex=====")
    # abalone = DataSet("../data/abalone.data", 0, regression=False)
    # runClassificationAlgorithms(abalone)

    # print("=====Car Classification of Acceptability=====")
    # car = DataSet("../data/car.data", target_location=6, isCars=True, regression=False)
    # runClassificationAlgorithms(car)

    # print("=====Image Classification based on Pixel Information=====")
    # segmentation = DataSet("../data/segmentation.data", target_location=0, regression=False)
    # runClassificationAlgorithms(segmentation)
    # print("=====Abalone Classification of Sex=====")
    abalone = DataSet("../data/abalone.data", 0, regression=False)
    ffn = FeedForwardNetwork(8, 1, [12], 3, learning_rate=.1, momentum_constant=.8)
    abalone.runAlgorithm(ffn)
    # #
    # cars = DataSet("../data/car.data", target_location=6, isCars=True, regression=False);
    # # best settings = FeedForwardNetwork(6, 0, [], 4, learning_rate=.00001, momentum_constant=0)
    # ffn = FeedForwardNetwork(6, 0, [], 4, learning_rate=.00001, momentum_constant=.1)
    # cars.runAlgorithm(ffn)
    # print("=====Wine Quality=====")
    # wine = DataSet("../data/winequality.data", target_location=11, regression=True)
    # # wine zero best: FeedForwardNetwork(11, 0, [], 1, learning_rate=.00001, momentum_constant=.2)
    # ffn = FeedForwardNetwork(11, 2, [11, 11], 1, learning_rate=0.001, momentum_constant=.2)
    # wine.runAlgorithm(ffn)


    # print("=====Forest Fire Regression (Area Burned (hectares))=====")
    # forest_fire = DataSet("../data/forestfires.data", target_location=12, dates=2, days=3, regression=True)
    # # # best for zero layer forest fire: FeedForwardNetwork(12, 0, [], 1, learning_rate=.00001, momentum_constant=.2)
    # # # best for 1 layer FeedForwardNetwork(12, 1, [12], 1, learning_rate=.00001, momentum_constant=.2)
    # ffn = FeedForwardNetwork(12, 2, [12, 12], 1, learning_rate=0.0005, momentum_constant=.3)
    # forest_fire.runAlgorithm(ffn)

    segmentation = DataSet("../data/segmentation.data", target_location=0, regression=False)
    # 19 features, testing 1 hidden layer, with 10 nodes, 7 class outputs
    # cnn, means, medoids = getCNNMeansAndMedoids(segmentation, 5)
    # runRadialBasisNetworkAlgorithms(cnn, means, medoids, segmentation, 19, 7, .001)
    # print("Initializing Radial Basis Network with CNN Set")
    ffn = FeedForwardNetwork(19, 2, [19, 19], 7, learning_rate=.1, momentum_constant=.01)
    segmentation.runAlgorithm(ffn)

   #  print("=====Machine Performance Regression (relative performance)=====")
   #  machine = DataSet("../data/machine.data", target_location=7, ignore=[0, 1], regression=True)
   #  # knn = KNearestNeighbor(5)
   # # machine.runAlgorithm(knn)
   #  ffn = FeedForwardNetwork(7, 2, [8, 8], 1, learning_rate=.001, momentum_constant=.2)
   #  machine.runAlgorithm(ffn)
    # runRegressionAlgorithms(machine)
    #
    # PAMClusterNew(segmentation, 100)
    #
    # KMeans(segmentation, 100)


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
# run the regression based algorithms
def runRegressionAlgorithms(dataset):
    k_values = [5,10,15]
    # running knn in respect to k value
    for i in k_values:
        print("Running KNN with K of {}".format(i))
        dataset.runAlgorithm(KNearestNeighbor(i))
    k = math.ceil(len(dataset.data) / 4)
    # running kmeans in respect to the dataset
    KMeans(dataset, k)
def runClassificationAlgorithms(dataset):
    k_values = [5, 10, 15]
    # running each algorithm in respect to each k value
    # for i in k_values:
    #     print("Running KNN with K of {}".format(i))
    #     dataset.runAlgorithm(KNearestNeighbor(i))
    # for i in k_values:
    #     print("Running CNN with K of {}".format(i))
    #     dataset.runAlgorithm(CondensedNearestNeighbor(i))
    # for i in k_values:
    #     print("Running ENN with K of {}".format(i))
    #     dataset.runAlgorithm(EditedNearestNeighbor(i))
    # running kmeans in respect to last ENN
    KMeans(dataset, 20)


if __name__ == '__main__':
    main()




#print("debug")