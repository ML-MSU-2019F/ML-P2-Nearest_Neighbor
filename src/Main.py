from DataSet import DataSet
from NearestNeigbor.KNearestNeighbor import KNearestNeighbor
from NearestNeigbor.PAM import PAMClusterNew
from NearestNeigbor.KMeans import KMeans
from FeedForwardNetwork import FeedForwardNetwork
import math
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
    # print("=====Machine Performance Regression (relative performance)=====")
    # machine = DataSet("../data/machine.data", target_location=7, ignore=[0, 1], regression=True)
    # runRegressionAlgorithms(machine)

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

    segmentation = DataSet("../data/segmentation.data", target_location=0, regression=False)
    # 19 features, testing 1 hidden layer, with 10 nodes, 7 class outputs
    ffn = FeedForwardNetwork(19, 1, [10], 7)
    segmentation.runAlgorithm(ffn)

    #
    # PAMClusterNew(segmentation, 100)
    #
    # KMeans(segmentation, 100)

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