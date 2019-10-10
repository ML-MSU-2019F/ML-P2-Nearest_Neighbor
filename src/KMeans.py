from EditedNearestNeighbor import EditedNearestNeighbor
from KNearestNeighbor import KNearestNeighbor
from DataSet import DataSet
import math
import numpy
class KMeans(KNearestNeighbor):
    def __init__(self, enn_result:DataSet, k_centers):
        self.data_set = enn_result
        data = enn_result.separateClassFromData()
        data = numpy.array(data)
        centroids = []
        # initialize based on first i indexes, which is random
        for i in range(0,k_centers):
            centroids.append(data[i])
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
                closest = self.getNearestNeighbor(one,all,1)
                data_of_closest = all[closest[0][2]]
                centroid_group = self.getChosenCentroid(centroids,data_of_closest)
                centroid_groups[centroid_group].append(line)

            for i in range(0,k_centers):
                # get the mean of all the values

                np_array = numpy.array(centroid_groups[i])
                mean = numpy.mean(np_array,axis=0)
                difference = abs(mean - np_array)
                movement += abs(numpy.sum(difference))
                centroids[i] = mean
            if movement == last_movement:
                break
            centroid_groups = []
            for i in range(0, k_centers):
                centroid_groups.append([])
            print("Movement was {}".format(movement))

        for i in range(0,len(centroids)):
            centroids[i] = numpy.insert(centroids[i], self.data_set.class_location, 0)

        list_centroids = []
        for i in range(0,len(centroids)):
            nearest = self.getNearestNeighbor(centroids[i], enn_result.data, 1)
            centroids[i] = numpy.delete(centroids[i],0)
            list_centroids.append(list(centroids[i]))
            list_centroids[i].insert(0, nearest[0][1])

        self.runTenFold(list_centroids)
        print(list_centroids)


    def getChosenCentroid(self,centroids,chosen):
        chosen = numpy.array(chosen)
        centroids = numpy.array(centroids)
        for i in range(0,len(centroids)):
            if numpy.array_equal(chosen,centroids[i]):
                return i
