from KNearestNeighbor import KNearestNeighbor
from DataSet import DataSet
import numpy
import random
import math


class PAMClusterNew(KNearestNeighbor):
    medoids = None

    def __init__(self, original: DataSet, medoid_count):
        # set up original, undedited passed dataset
        self.data_set = original
        data = None
        # if the algo result is there, use that instead
        if len(original.algo_result) is not 0:
            data = original.algo_result
        else:
            data = original.data
        self.data = data
        # If its not regression, we need to separate the data from the class so that we can use numpy
        if not self.data_set.regression:
            data = original.separateClassFromData()
        # initialize variable data as a numpy array of the data
        data = numpy.array(data)

        medoids = []
        # simply makes sure we don't select the same random twice
        medoid_indexes = []
        # get initial medoids based on random selection from data
        while len(medoids) != medoid_count:
            rand = random.randint(0, len(data) - 1)
            if rand in medoid_indexes:
                continue
            medoid_indexes.append(rand)
            medoids.append(data[rand])

        medoid_groups = []
        for i in range(0, medoid_count):
            medoid_groups.append([])

        # keep track if the medoids changed
        medoids_changed = True
        loops = 0
        while medoids_changed:
            print("Iteration {}".format(loops))
            loops += 1
            medoids_changed = False
            # get all closest neighbors of medoids
            for line in data:
                # get the closest medoid to line
                closest = self.getNearestNeighbor(line, medoids, 1, skip_class=False)[0]
                # closest[2] = index of medoid within the medoids list
                medoid_groups[closest[2]].append(line)
            # get the original distortion to compare to the swapped distortions
            medoid_original_distortion = []
            for i in range(0, len(medoids)):
                # make the index availble for assignment
                medoid_original_distortion.append([])
                # get the distortion between the medoid and its group of lines closest to it
                medoid_original_distortion[i] = self.getDistortion(medoids[i], medoid_groups[i])
            # go through our medoids,
            for i in range(0, len(medoids)):
                distortion = None
                # set original minimum distortion equal to the distortion from original medoid
                min_medoid_distortion = medoid_original_distortion[i]
                # set the original min medoid equal to the original medoid
                min_medoid = medoids[i]
                # go through the lines relating to the medoid groups
                for j in range(0, len(medoid_groups[i])):
                    # get the single item in group
                    single = medoid_groups[i][j]
                    # get all the other items
                    multi = medoid_groups[i][:j] + medoid_groups[i][j+1:]
                    # get the distortion between the single, and other items
                    new_distortion = self.getDistortion(single, multi)
                    # if our new distortion is less than our current min, set the mins to our current single, and the
                    # distortion to our current max
                    if new_distortion < min_medoid_distortion:
                        min_medoid_distortion = new_distortion
                        min_medoid = single
                # the arrays would be equal if there was no change, if we have even a single medoid change, mark as
                # changes and rerun the loop
                if not numpy.array_equal(min_medoid, medoids[i]):
                    medoids_changed = True
                # set the new medoid equal to our min medoid that we got from the loop above
                medoids[i] = min_medoid
        # loop is finished, find the original class of the medoid by getting the matching data element
        for i in range(0, len(medoids)):
            medoids[i] = medoids[i].tolist()
            # insert the class back into the data
            medoids[i].insert(self.data_set.target_location, self.getOriginalClass(medoids[i]))
        # check the accuracy of our medoids against the original data set
        print(self.checkAccuracyAgainstSet(medoids, self.data, 1))

    def getOriginalClass(self, line):
        location = self.data_set.target_location;
        for orig_line in self.data:
            compare_line = numpy.array(orig_line[:location] + orig_line[location+1:])
            if numpy.array_equal(compare_line, line):
                print("found original class!")
                return orig_line[self.data_set.target_location]

    def getDistortion(self,single,multi):
        distortion = 0
        for i in range(0, len(multi)):
            distortion += self.getDistance(single, multi[i])
        return distortion


    def getDistance(self, one, two):
        one = numpy.array(one)
        two = numpy.array(two)
        sub = numpy.subtract(one, two)
        numpy.power(sub, 2)
        return numpy.sum(sub)

    def getChosenMedoid(self, centroids, chosen):
        chosen = numpy.array(chosen)
        centroids = numpy.array(centroids)
        for i in range(0, len(centroids)):
            if numpy.array_equal(chosen,centroids[i]):
                return i
