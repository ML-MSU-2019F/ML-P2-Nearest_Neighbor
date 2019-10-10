import heapq
import multiprocessing
import numpy
from DataSet import DataSet
from NearestNeighbor import NearestNeighbor
class KNearestNeighbor(NearestNeighbor):
    # shared data_set object
    data_set: DataSet = None
    # k nearest neighbors
    k = None

    def __init__(self, k):
        self.k = k

    def run(self,data_set: DataSet):
        self.data_set = data_set
        fold_validation = 10
        data_set.makeRandomMap(fold_validation)
        # go through each split
        results = multiprocessing.Array('i',[0]*2)
        process_array = []
        for i in range(0,fold_validation):
            p = multiprocessing.Process(target=self.runTenFold,args=(i, results))
            process_array.append(p)
            p.start()
        for process in process_array:
            process.join()
        print("Accuracy {:2.2f}".format((results[1]/results[0]) * 100))

    def runTenFold(self,num,results):
        one = self.data_set.getRandomMap(num)
        all = self.data_set.getAllRandomExcept(num)
        results[0] += len(one)
        # check all lines in test set against training set
        local_result = 0
        for line in one:
            closest = self.getNearestNeighbor(line, all)
            # classify 1 if correct, 0 if incorrect
            local_result += self.classify(line[self.data_set.class_location],closest)
        results[1] += local_result

    def getNearestNeighbor(self,line,train_set):
        # Iterate through all values in the training set
        distance_array = []
        for index in range(0, len(train_set)):
            # Keep track of the distance away from all points
            distance_array.append([])
            # check how far away test_line is from line
            for i in range(0, len(line)):
                # skip class index
                if i is self.data_set.class_location:
                    continue
                float_1 = float(line[i])
                float_2 = float(train_set[index][i])
                raw_distance = float_1 - float_2
                distance_array[index].append((raw_distance))
        # init raw distances into numpy array
        nd_array = numpy.array(distance_array)
        # Square each distance
        new_distance = numpy.power(nd_array, 2)
        # take the sum of distances across rows
        arr = numpy.sum(new_distance, 1)
        # turn arr into a tuple array so we can heap the data and get the minimums with class information
        tuple_arr = []
        for i in range(0, len(arr)):
            tuple_arr.append((arr[i], train_set[i][self.data_set.class_location]))
        # turn the tuple into a heap O(nlgn)
        heapq.heapify(tuple_arr)

        # Below This will most likely get separated into a classify function
        # pop off k smallest values
        closest = heapq.nsmallest(self.k, tuple_arr)
        return closest

    def classify(self,actual_class,closest):
        occurrence_dict = {}
        # Add up class occurrences
        for i in range(0, len(closest)):
            if closest[i][1] not in occurrence_dict:
                occurrence_dict[closest[i][1]] = 1
            else:
                occurrence_dict[closest[i][1]] += 1
        # find the max occurrence of class from the count before
        max_class_num = -1
        max_class = ""
        for class_key in occurrence_dict:
            occurrences = occurrence_dict[class_key]
            if occurrences > max_class_num:
                max_class = class_key
                max_class_num = occurrences
        # if our max occurrence was the same as the actual class, return a 1
        if max_class == actual_class:
            return 1
        # else return a 0
        else:
            return 0
