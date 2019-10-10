from DataSet import DataSet
from KNearestNeighbor import KNearestNeighbor
from EditedNearestNeighbor import EditedNearestNeighbor
x_fold_validation = 10
def main():

    # abalone = DataSet("../data/abalone.data", 0)
    # knn = KNearestNeighbor(5)
    # abalone.runAlgorithm(knn)

    # car = DataSet("../data/car.data",0)
    # enn = EditedNearestNeighbor(5)
    # car.runAlgorithm(enn)

    segmentation = DataSet("../data/segmentation.data", 0)
    knn = KNearestNeighbor(5)
    enn = EditedNearestNeighbor(5)
    segmentation.runAlgorithm(knn)
    segmentation.runAlgorithm(enn)

if __name__ == '__main__':
    main()




#print("debug")