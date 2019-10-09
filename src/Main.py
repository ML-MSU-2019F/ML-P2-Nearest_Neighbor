from DataSet import DataSet
from KNearestNeighbor import KNearestNeighbor
x_fold_validation = 10
def main():

    abalone = DataSet("../data/abalone.data", 0)
    knn = KNearestNeighbor(5)
    abalone.runAlgorithm(knn)

    segmentation = DataSet("../data/segmentation.data", 0)
    knn = KNearestNeighbor(5)
    segmentation.runAlgorithm(knn)

if __name__ == '__main__':
    main()




#print("debug")