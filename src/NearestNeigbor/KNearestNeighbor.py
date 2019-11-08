import heapq
import multiprocessing
import numpy
from Algorithm import Algorithm


class KNearestNeighbor(Algorithm):
    # shared data_set object
    # k nearest neighbors
    k = None
    regression = None

    # pass in k for how many neighbors by default we will be using
    def __init__(self, k, data_set = None):
        self.k = k
        self.data_set = data_set

    # the run function for KNN, other NN algorithms will make their own.
    # a processed data_set is passed, and whether or not the algorithm will regress
    # or classify
    def run(self, data_set, regression):
        # set class variables
        self.data_set = data_set
        self.regression = regression
        # run ten fold
        results = self.runTenFold(self.data_set.data)
        # if not regression, print out in terms of 0-1 loss percentage
        if not regression:
            print("Accuracy {:2.2f}%".format((results[1] / results[0]) * 100))
        else:
            print("MAE {:2.2f}".format((results[1] / results[0])))

    # runTenFold takes in data you pass it, and runs ten fold validation on that set using 10 spawned processes
    # multiple processes are used because this is very slow on a single core
    def runTenFold(self,data):
        # initialize a random map that will get used by the processes
        self.data_set.makeRandomMap(data, 10)
        # make a multiprocessing array so that we can hold results from processes
        results = multiprocessing.Array('f', [0] * 2)
        # process array to hold all processes and halt us until all finish
        process_array = []
        # for 10 folds
        for i in range(0, 10):
            # spawn new process
            p = multiprocessing.Process(target=self.foldProcess, args=(i, results))
            process_array.append(p)
            p.start()
        # wait for all processes to finish
        for process in process_array:
            process.join()
            process.kill()
        # return the total results
        return results

    # fold process it the act of doing one fold in ten fold validation
    def foldProcess(self,num,results):
        # 1/10 random
        one = self.data_set.getRandomMap(num)
        # 9/10 random
        all = self.data_set.getAllRandomExcept(num)
        # this is the amount of samples we  will be running
        results[0] += len(one)
        # hold local result anr write once to result[1] to prevent process locking
        local_result = 0
        # check all lines in test set against training set
        for line in one:
            # get 5 closest to the line
            closest = self.getNearestNeighbor(line, all, 5)
            # grab actual value
            actual_value = line[self.data_set.target_location]
            # if not regression
            if not self.regression:
                # get classification
                local_result += self.classify(actual_value, closest)
            else:
                #get regression
                local_result += self.regress(actual_value, closest)
        # add local results to total results
        results[1] += local_result

    # get the nearest neighbor given a point, a set and k
    def getNearestNeighbor(self,line,train_set,k=None,skip_class=True):
        # set k to self.k
        k_local = self.k
        # unless k is specified as an argument
        if k is not None:
            k_local = k
        # keep a raw distance array, so we can use matrix operations and numpy to make things quick
        distance_array = []
        # Iterate through all values in the training set
        for index in range(0, len(train_set)):
            # Keep track of the distance away from all points
            distance_array.append([])
            # check how far away test_line is from line
            for i in range(0, len(line)):
                # skip class index
                distance = None
                # if we are at the target, or what we are trying to predict, skip
                if skip_class:
                    if i is self.data_set.target_location:
                        continue
                # if we are on a day index, get the difference between the two days
                if i is self.data_set.day_index:
                    distance = self.data_set.getDayDifference(line[i],train_set[index][i])
                # if we are on a date index, get the difference in dates
                elif i is self.data_set.date_index:
                    distance = self.data_set.getDateDifference(line[i],train_set[index][i])
                else:
                    # otherwise get the float distance between the two points
                    float_1 = float(line[i])
                    float_2 = float(train_set[index][i])
                    distance = float_1 - float_2
                # append the distance to the distance array
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
            # in the case that our location has been stripped, return placeholder of 0
            if location < len(train_set[i]):
                tuple_arr.append((arr[i], train_set[i][location],i))
            else:
                tuple_arr.append((arr[i], 0, i))
        # turn the tuple into a heap
        heapq.heapify(tuple_arr)
        # pop off k smallest values and return them
        closest = heapq.nsmallest(k_local, tuple_arr)
        return closest

    def checkAccuracyAgainstSet(self,classify_data,test_set,k=5):
        result = 0
        for line in test_set:
            closest = self.getNearestNeighbor(line,classify_data,k)
            actual = line[self.data_set.target_location]
            if self.data_set.regression:
                result += self.regress(actual,closest)
            else:
                result += self.classify(actual, closest)
        return result/len(test_set)

    # regress function based on closest values and the actual value
    # the mean is taken from the closest values and used to predict
    # actual then the Mean Absolute error is given back as a return
    def regress(self,actual_value,closest_values):
        total = 0
        for i in range(0,len(closest_values)):
            total += numpy.power(float(closest_values[i][1]),2)
        mean = total/len(closest_values)
        return self.MAE(float(actual_value),mean)

    # mean absolute error, get distance between actual and predicted
    # and absolute value it
    def MAE(self,actual,predicted):
        return abs(actual - predicted)

    # classify based on closest classes, and actual class
    def classify(self, actual_class, closest):
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
