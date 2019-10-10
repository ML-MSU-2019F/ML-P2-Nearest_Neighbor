from DataSet import DataSet
from KNearestNeighbor import KNearestNeighbor
from EditedNearestNeighbor import EditedNearestNeighbor
from CondensedNearestNeighbor import CondensedNearestNeighbor
from KMeans import KMeans
def main():

    # abalone = DataSet("../data/abalone.data", 0)
    # enn = EditedNearestNeighbor(5)
    # abalone.runAlgorithm(enn)

    # car = DataSet("../data/car.data",0)
    # enn = EditedNearestNeighbor(5)
    # car.runAlgorithm(enn)

    segmentation = DataSet("../data/segmentation.data", 0)
    cnn = CondensedNearestNeighbor(5)
    enn = EditedNearestNeighbor(5)
   # print("Starting Condensed Nearest Neighbor")
   # print("-----------------------------------")
   # segmentation.runAlgorithm(cnn)
    print("Starting Edited Nearest Neighbor")
    print("-----------------------------------")
    # segmentation.runAlgorithm(enn)
    kmeans = KMeans(segmentation,5)
    # knn = KNearestNeighbor(5)
    #segmentation.runAlgorithm(knn)
    #segmentation.runAlgorithm(knn)

if __name__ == '__main__':
    main()




#print("debug")