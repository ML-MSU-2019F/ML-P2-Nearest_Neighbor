import random
import Algorithm
import numpy
from ReadFile import ReadFile


class DataSet():
    data = None
    file_path = None
    target_location = None
    date_values = {"jan":1, "feb":2, "mar":3, "apr":4, "may":5, "jun":6, "jul":7, "aug":8, "sep":9, "oct":10, "nov":11, "dec":12}
    day_values = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5, "sat": 6, "sun": 7}
    low_high_values = {"vhigh": 20, "high":10, "med":5, "low": 0, "small": 0, "big": 10, "more": 6, "5more": 6}
    date_index = None
    day_index = None
    algo_result = []
    regression = None

    def __init__(self, file_path, target_location = None, regression = None, dates=None, days = None, predict = None, missing_value=None, columns=None, isCars=False, ignore=None):
        self.target_location = target_location
        self.file_path = file_path
        self.regression = regression
        self.date_index = dates
        self.days_index = days
        self.data = ReadFile.read(self,file_path)
        if ignore is not None:
            self.data = self.removeColumns(ignore)
        if isCars:
            for i in range(0,len(self.data)):
                for j in range(0,len(self.data[0])):
                    value = self.data[i][j]
                    if value in self.low_high_values:
                        self.data[i][j] = self.low_high_values[value]
        if dates is not None:
            for i in range(0,len(self.data)):
                self.data[i][dates] = self.date_values[self.data[i][dates]]
        if days is not None:
            for i in range(0,len(self.data)):
                self.data[i][days] = self.day_values[self.data[i][days]]
        if regression:
            for i in range(0,len(self.data)):
                for j in range(0,len(self.data[0])):
                    self.data[i][j] = float(self.data[i][j])
        if target_location is not None and not regression:
            raw_data = self.separateClassFromData()
            np_array = numpy.array(raw_data)
            np_array = (np_array - np_array.min()) / (np_array.max() - np_array.min()                                       )
            self.data = self.joinClassAndData(np_array)

    def runAlgorithm(self, algorithm: Algorithm):
        if self.regression is not None:
            algorithm.run(self,self.regression)
        else:
            algorithm.run(self)

    def makeRandomMap(self, data, bins):
        temp_data = data[0:]
        random_map = []
        current_bin = 0
        for i in range(0, bins):
            random_map.append([])
        while len(temp_data) is not 0:
            random_choice = random.randint(0, len(temp_data) - 1)
            random_map[current_bin].append(temp_data[random_choice])
            del temp_data[random_choice]
            current_bin = current_bin + 1
            if (current_bin >= bins):
                current_bin = 0
        self.random_map = random_map

    def getRandomMap(self,index):
        return self.random_map[index]
    def getAllRandomExcept(self,remove_index):
        result = []
        for index in range(0,len(self.random_map) - 1):
            if remove_index is index:
                continue
            else:
                for rows in self.random_map[index]:
                    result.append(rows)
        return result

    def getRandomMap(self, index):
        return self.random_map[index]

    # Utility Functions

    def separateClassFromData(self):
        result = []
        for i in range(0,len(self.data)):
            result.append([])
            for j in range(0,len(self.data[i])):
                if j is self.target_location:continue
                result[i].append(float(self.data[i][j]))
        return result

    def joinClassAndData(self, data):
        result = []
        for i in range(0,len(data)):
            result.append([])
            for j in range(0,len(data[i])):
                if j is self.target_location:
                    result[i].append(self.data[i][self.target_location])
                result[i].append(data[i][j])
            if len(data[i]) is self.target_location:
                result[i].append(self.data[i][self.target_location])
        return result

    def getDateDifference(self,term1,term2):
        # normalize through / len of values
        return self.getRollingDifference(12, term1, term2)/len(self.date_values)

    def getDayDifference(self,term1,term2):
        return self.getRollingDifference(7,term1,term2)/len(self.day_values)

    def getRollingDifference(self,length,term1,term2):
        if term1 > term2:
            temp = term1
            term1 = term2
            term2 = temp
        distance1 = term2 - term1
        distance2 = length - distance1
        return min(distance1, distance2)

    def removeColumns(self,columns):
        map = {}
        for i in columns:
            map[i] = True
        result = []
        for i in range(0, len(self.data)):
            result.append([])
            for j in range(0, len(self.data[i])):
                if j in map:
                    continue
                else:
                    result[i].append(self.data[i][j])
        return result