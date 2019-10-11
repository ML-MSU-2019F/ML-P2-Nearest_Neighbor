from EditedNearestNeighbor import EditedNearestNeighbor
from KNearestNeighbor import KNearestNeighbor
from DataSet import DataSet
import multiprocessing
import math
import numpy
import random
class KMeans(KNearestNeighbor):
    centroids = None
    def __init__(self, enn_result:DataSet, k_centers):
        self.data_set = enn_result
        if len(enn_result.algo_result) is not 0:
            data = enn_result.algo_result
        else:
            data = enn_result.data
        self.data = data
        if not self.data_set.regression:
            data = enn_result.separateClassFromData()

        data = numpy.array(data)
        centroids = []
        # initialize based on first i indexes, which is random
        for i in range(0,k_centers):
            rand = random.randint(0,len(data)-1)
            centroids.append(data[rand])
        centroid_groups = []
        for i in range(0, k_centers):
            centroid_groups.append([])

        movement = math.inf
        while movement > .1:
            last_movement = movement
            movement = 0
            for line in data:
                one = line
                all = centroids
                # get nearest centroid
                closest = self.getNearestNeighbor(one,all,1)[0]
                data_of_closest = all[closest[2]]
                centroid_group = self.getChosenCentroid(centroids,data_of_closest)
                centroid_groups[centroid_group].append(line)

            for i in range(0,k_centers):
                # get the mean of all the values
                if len(centroid_groups[i]) is 0:
                    continue
                np_array = numpy.array(centroid_groups[i])
                mean = numpy.mean(np_array,axis=0)
                difference = numpy.setdiff1d(mean,np_array)
                movement += numpy.sum(difference)
                centroids[i] = mean
            if movement == last_movement:
                break
            centroid_groups = []
            for i in range(0, k_centers):
                centroid_groups.append([])
            print("Movement was {}".format(movement))

        list_centroids = []
        location = self.data_set.target_location
        for i in range(0,len(centroids)):
            nearest = self.getNearestNeighbor(centroids[i], enn_result.data, 1)
            list_centroids.append([])
            list_centroids[i] = centroids[i].tolist()
            list_centroids[i].insert(location,nearest[0][1])
        self.centroids = list_centroids
        self.getAccuracy()

    def getAccuracy(self):
        results = 0
        for line in self.data:
            closest = self.getNearestNeighbor(line,self.centroids,1)
            if not self.data_set.regression:
                results += self.classify(line[self.data_set.target_location], closest)
            else:
                results += self.regress(line[self.data_set.target_location], closest)
        if not self.data_set.regression:
            print("Accuracy was: {:2.2f}".format((results / len(self.data_set.data)) * 100))
        else:
            print("MSE was: {:2.2f}".format(results / len(self.data_set.data)))


    def getChosenCentroid(self,centroids,chosen):
        chosen = numpy.array(chosen)
        centroids = numpy.array(centroids)
        for i in range(0,len(centroids)):
            if numpy.array_equal(chosen,centroids[i]):
                return i
