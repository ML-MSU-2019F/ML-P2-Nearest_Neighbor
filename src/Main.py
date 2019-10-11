from DataSet import DataSet
from KNearestNeighbor import KNearestNeighbor
from EditedNearestNeighbor import EditedNearestNeighbor
from CondensedNearestNeighbor import CondensedNearestNeighbor
from KMeans import KMeans
import math
def main():

    # abalone = DataSet("../data/abalone.data", 0)
    # enn = EditedNearestNeighbor(5)
    # abalone.runAlgorithm(enn)
    # print("=====Wine Quality=====")
    # wine = DataSet("../data/winequality.data", target_location=11, regression=True)
    # knn = KNearestNeighbor(5)
    # wine.runAlgorithm(knn)
    # segmentation = DataSet("../data/segmentation.data", 0)
    # kmeans = KMeans(segmentation, 50)

    # print("=====Forest Fire Regression (Area Burned (hectares))=====")
    # forest_fire = DataSet("../data/forestfires.data", target_location=12, dates=2, days=3, regression=True)
    # knn = KNearestNeighbor(5)
    # forest_fire.runAlgorithm(knn)
    # kmeans = KMeans(forest_fire, math.ceil(len(forest_fire.data)/4))
    #
    # print("=====Machine Performance Regression (relative performance)=====")
    # machine = DataSet("../data/machine.data", target_location=7, ignore=[0, 1], regression=True)
    # knn = KNearestNeighbor(5)
    # machine.runAlgorithm(knn)
    # kmeans = KMeans(forest_fire, math.ceil(len(machine.data) / 4))



    # TODO: Rename target_location to a word that works for both regression and classification
    car = DataSet("../data/car.data",target_location=6,isCars=True, regression=False)
    #knn = KNearestNeighbor(5)
    enn = EditedNearestNeighbor(5)
    cnn = CondensedNearestNeighbor(5)
    #car.runAlgorithm(cnn)
    car.runAlgorithm(enn)
    #car.runAlgorithm(knn)
    kmeans = KMeans(car,3)
    # enn = EditedNearestNeighbor(5)
    # car.runAlgorithm(enn)
    # cnn = CondensedNearestNeighbor(5)
    #enn = EditedNearestNeighbor(5)
   # print("Starting Condensed Nearest Neighbor")
   # print("-----------------------------------")
    #segmentation.runAlgorithm(enn)
    #print("Starting Edited Nearest Neighbor")
    #print("-----------------------------------")
  #  segmentation.runAlgorithm(enn)
    #kmeans = KMeans(segmentation,3)
    # knn = KNearestNeighbor(5)
    #segmentation.runAlgorithm(knn)
    #segmentation.runAlgorithm(knn)

if __name__ == '__main__':
    main()




#print("debug")