from KNearestNeighbor import KNearestNeighbor
from DataSet import DataSet
import numpy
import random
import math


class PAMClusterNew(KNearestNeighbor):
    medoids = None

    def __init__(self, original: DataSet, medoid_count):
        self.data_set = original
        data = None
        if len(original.algo_result) is not 0:
            data = original.algo_result
        else:
            data = original.data
        self.data = data

        if not self.data_set.regression:
            data = original.separateClassFromData()

        data = numpy.array(data)

        medoids = []
        medoid_indexes = []
        # get the initial medoids
        while len(medoid_indexes) != medoid_count:
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
        while medoids_changed and loops < 50:
            print("Iteration {}".format(loops))
            loops += 1
            medoids_changed = False
            # get all closest neighbors of medoids
            for line in data:
                one = line
                # get the closest nearest centroid
                closest = self.getNearestNeighbor(one, medoids, 1, skip_class=False)[0]
                data_of_closest = medoids[closest[2]]
                # get what centroid group to add line to and add it
                medoid_group = self.getChosenMedoid(medoids, data_of_closest)
                medoid_groups[medoid_group].append(line)
            medoid_original_distortion = []
            for i in range(0, len(medoids)):
                medoid_original_distortion.append([])
                medoid_original_distortion[i] = self.getDistortion(medoids[i], medoid_groups[i])

            for i in range(0, len(medoids)):
                distortion = None
                min_medoid_distortion = medoid_original_distortion[i]
                min_medoid = medoids[i]
                for j in range(0, len(medoid_groups[i])):
                    single = medoid_groups[i][j]
                    multi = medoid_groups[i][:j] + medoid_groups[i][j+1:]
                    new_distortion = self.getDistortion(single, multi)
                    if new_distortion < min_medoid_distortion:
                        min_medoid_distortion = new_distortion
                        min_medoid = single
                # track if we had a medoid change
                if not numpy.array_equal(min_medoid, medoids[i]):
                    medoids_changed = True
                medoids[i] = min_medoid
        print("here we are")
        for i in range(0, len(medoids)):
            print(self.getOriginalClass(medoids[i]))
            medoids[i] = medoids[i].tolist();
            medoids[i].insert(self.data_set.target_location, self.getOriginalClass(medoids[i]));
        print(medoids)
        print(self.checkAccuracyAgainstSet(medoids, self.data, 1));

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
