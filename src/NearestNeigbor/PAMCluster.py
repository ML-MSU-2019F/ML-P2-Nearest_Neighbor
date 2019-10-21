from NearestNeigbor.KNearestNeighbor import KNearestNeighbor
from DataSet import DataSet
import math
import numpy
import random
class PAMCluster(KNearestNeighbor):
    medoids = None
    def __init__(self,enn_result:DataSet):
        self.data_set = enn_result
        if len(enn_result.algo_result) is not 0:
            data = enn_result.algo_result
        else:
            data = enn_result.data
        self.data = data
        if not self.data_set.regression:
            data = enn_result.separateClassFromData()

        data = numpy.array(data)
        medoids = []

        # initialize randomly selected k medoid(s) (enn_result) as data points in n
        for i in range(0, enn_result):
            rand = random.randint(0, len(data) - 1)
            medoids.append(data[rand])
        medoid_groups = []
        for i in range(0, enn_result):
            medoid_groups.append([])
        #associate each point with closest medoid
        for line in data:
            one = line
            all = medoids
            # get nearest medoid
            closest = self.getNearestNeighbor(one, all, 1)[0]
            data_of_closest = all[closest[2]]
            medoid_group = self.getChosenMedoid(medoids, data_of_closest)
            medoid_groups[medoid_group].append(line)
            #calculate current cost
        #for each m medoid calculate dissimilarity of each node then calculate the cost
        cost =math.inf
        for i in range(0,enn_result):
            if len(medoid_groups[i]) is 0:
                continue
                np_array = numpy.array(medoid_groups[i])
                difference = numpy.setdiff1d(medoid[i],np_array)
                cost += numpy.sum(difference)
            print("Initial cost was {}".format(difference))

        last_cost = cost
        #while cost difference is greater than 0,    take m medoid and swap it with o point within the group. using smallest dissimilarity
        while (last_cost > 0):
            for o in range(0, enn_result):
                if len(medoid_groups[i]) is 0:
                    continue
                #make sure m != o








    def getChosenMedoid(self,medoids,chosen):
        chosen = numpy.array(chosen)
        medoids = numpy.array(medoids)
        for i in range(0,len(medoids)):
            if numpy.array_equal(chosen,medoids[i]):
                return i