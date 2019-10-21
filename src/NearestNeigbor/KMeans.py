from NearestNeigbor.KNearestNeighbor import KNearestNeighbor
from DataSet import DataSet
import numpy


class KMeans(KNearestNeighbor):
    centroids = None

    def __init__(self, original:DataSet, k_centers):
        # initialize data, if algo was done previously on original, use it as data
        self.data_set = original
        data = None
        if len(original.algo_result) is not 0:
            data = original.algo_result
        else:
            data = original.data
        # set class variable for data
        self.data = data[0:]
        # if its not regression, separate our class from our data when forming clusters
        if not self.data_set.regression:
            data = original.separateClassFromData()

        data = numpy.array(data)
        # holds actual centroids
        centroids = []
        # initialize based on first i indexes, which is random
        self.data_set.makeRandomMap(data.tolist(),1)
        rand_map = self.data_set.getRandomMap(0)[0:]
        for i in range(0, k_centers):
            centroids.append(rand_map[i])
        # holds data which has a closest neighbor of a centroid
        centroid_groups = []
        for i in range(0, k_centers):
            centroid_groups.append([])
        # move while its significant
        last_means = None
        while True:
            print("Moving")
            # go through the lines in our data
            for line in data:
                # get the closest nearest centroid, do not skip class in distance measures
                closest = self.getNearestNeighbor(line,centroids,1,skip_class=False)[0]
                # closest[2] = index of line within centroids
                centroid_groups[closest[2]].append(line)
            # go through our centroid groups
            means = numpy.array([])
            for i in range(0, k_centers):
                # get the mean of all the values
                np_array = numpy.array(centroid_groups[i])
                mean = numpy.mean(np_array, axis=0)
                means = numpy.append(means, mean)
                # adjust centroid group to be the mean of the distances
                centroids[i] = mean
            # if we didn't move from last time we are finished
            if numpy.array_equal(means, last_means):
                print("did not move, ending")
                break
            last_means = numpy.copy(means)
            # reset centroid groups
            centroid_groups = []
            for i in range(0, k_centers):
                centroid_groups.append([])
        # turn numpy array into regular list, with class variables/target variables put back
        list_centroids = []
        location = self.data_set.target_location
        for i in range(0, len(centroids)):
            # shift over at the class location, this prevents getting the nearest neighbor of data from the wrong
            # features
            centroids[i] = numpy.insert(centroids[i], location, 0)
            # get the nearest neighbor to the mean, and assume that is the class of the mean
            nearest = self.getNearestNeighbor(centroids[i], self.data, 1)
            # remove the shift
            centroids[i] = numpy.delete(centroids[i], location)
            # and insert the class into the location of the shift
            list_centroids.append([])
            list_centroids[i] = centroids[i].tolist()
            list_centroids[i].insert(location, nearest[0][1])
        # set the result centroids
        self.centroids = list_centroids
        # see how good it was
        # check the accuracy of the centroids vs the original data
        print(self.checkAccuracyAgainstSet(list_centroids, self.data, 1))

    # get the index of the centroid that was chosen by getting the closest nearest neighbor
    def getChosenCentroid(self, centroids, chosen):
        chosen = numpy.array(chosen)
        centroids = numpy.array(centroids)
        for i in range(0,len(centroids)):
            if numpy.array_equal(chosen, centroids[i]):
                return i
