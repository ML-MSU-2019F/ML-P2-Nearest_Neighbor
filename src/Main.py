from DataSet import DataSet
from KNearestNeighbor import KNearestNeighbor
from EditedNearestNeighbor import EditedNearestNeighbor
from CondensedNearestNeighbor import CondensedNearestNeighbor

def main():

    # abalone = DataSet("../data/abalone.data", 0)
    # enn = EditedNearestNeighbor(5)
    # abalone.runAlgorithm(enn)

    # car = DataSet("../data/car.data",0)
    # enn = EditedNearestNeighbor(5)
    # car.runAlgorithm(enn)

    segmentation = DataSet("../data/segmentation.data", 0)
    cnn = CondensedNearestNeighbor(5)
    segmentation.runAlgorithm(cnn)
    # knn = KNearestNeighbor(5)
    #segmentation.runAlgorithm(knn)
    #segmentation.runAlgorithm(knn)

if __name__ == '__main__':
    main()




#print("debug")