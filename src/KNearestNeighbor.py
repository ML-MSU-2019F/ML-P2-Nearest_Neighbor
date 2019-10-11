import heapq
import multiprocessing
import numpy
from DataSet import DataSet
from NearestNeighbor import NearestNeighbor
class KNearestNeighbor(NearestNeighbor):
    # shared data_set object
    # k nearest neighbors
    k = None
    regression = None
    def __init__(self, k):
        self.k = k

    def run(self, data_set, regression):
        self.data_set = data_set
        self.regression = regression
        results = self.runTenFold(self.data_set.data)
        print(results[0])
        print(results[1])
        if not regression:
            print("Accuracy {:2.2f}".format((results[1] / results[0]) * 100))
        else:
            print("MSE {:2.2f}".format((results[1] / results[0])))

        # go through each split

    def runTenFold(self,data):
        self.data_set.makeRandomMap(data, 10)
        results = multiprocessing.Array('f', [0] * 2)
        results[1] = float(results[1])
        process_array = []
        for i in range(0, 10):
            p = multiprocessing.Process(target=self.foldProcess, args=(i, results))
            process_array.append(p)
            p.start()
        for process in process_array:
            process.join()
            process.kill()
        return results

    def foldProcess(self,num,results):
        one = self.data_set.getRandomMap(num)
        all = self.data_set.getAllRandomExcept(num)
        results[0] += len(one)
        # check all lines in test set against training set
        local_result = 0
        for line in one:
            closest = self.getNearestNeighbor(line, all, 5)
            # classify 1 if correct, 0 if incorrect
            actual_value = line[self.data_set.target_location]
            if not self.regression:
                local_result += self.classify(actual_value, closest)
            else:
                local_result += self.regress(actual_value, closest)
        results[1] += local_result

    def getNearestNeighbor(self,line,train_set,k=None):
        # Iterate through all values in the training set
        k_local = self.k
        if k is not None:
            k_local = k
        distance_array = []
        for index in range(0, len(train_set)):
            # Keep track of the distance away from all points
            distance_array.append([])
            # check how far away test_line is from line
            for i in range(0, len(line)):
                # skip class index
                distance = None
                if i is self.data_set.target_location:
                    continue
                if i is self.data_set.day_index:
                    distance = self.data_set.getDayDifference(line[i],train_set[index][i])
                elif i is self.data_set.date_index:
                    distance = self.data_set.getDateDifference(line[i],train_set[index][i])
                else:
                    float_1 = float(line[i])
                    float_2 = float(train_set[index][i])
                    distance = float_1 - float_2
                distance_array[index].append(distance)
        # init raw distances into numpy array
        nd_array = numpy.array(distance_array)
        # Square each distance
        new_distance = numpy.power(nd_array, 2)
        # take the sum of distances across rows
        arr = numpy.sum(new_distance, 1)
        # turn arr into a tuple array so we can heap the data and get the minimums with class information
        tuple_arr = []
        location = self.data_set.target_location
        for i in range(0, len(arr)):
            if location < len(train_set[i]):
                tuple_arr.append((arr[i], train_set[i][self.data_set.target_location],i))
            else:
                tuple_arr.append((arr[i], 0, i))
        # turn the tuple into a heap O(nlgn)
        heapq.heapify(tuple_arr)

        # Below This will most likely get separated into a classify function
        # pop off k smallest values
        closest = heapq.nsmallest(k_local, tuple_arr)
        return closest
    def regress(self,actual_value,closest_values):
        total = 0
        for i in range(0,len(closest_values)):
            total += float(closest_values[i][1])
        mean = total/len(closest_values)
        return self.MAE(float(actual_value),mean)

    def MAE(self,actual,predicted):
        return abs(actual - predicted)
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