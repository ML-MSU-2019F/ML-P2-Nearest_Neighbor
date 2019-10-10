import random
import Algorithm
import numpy
from ReadFile import ReadFile


class DataSet():
    data = None
    file_path = None
    class_location = None
    continuous_map = []
    class_array = []
    data_array = []
    def __init__(self, file_path, class_location = None, missing_value=None, columns=None):
        self.class_location = class_location
        self.file_path = file_path
        self.data = ReadFile.read(self,file_path)
        self.data = numpy.array(self.data)
        hasCategorical = self.generateNumericMap()
        if hasCategorical:
            for i in range(0,len(self.continuous_map)):
                if(not self.continuous_map[i]):
                    count = self.getCountInColumn(i)
        if class_location is not None:
            pure_data = self.separateClassFromData()
            np_array = numpy.array(pure_data)
            np_array = np_array / np_array.max(axis=0)
            self.data = self.joinClassAndData(np_array)
            print()
        if missing_value is not None and columns is not None:
            self.imputeData(missing_value, columns)

    def getCountInColumn(self,column):
        count_dict = {}
        for i in range(0,len(self.data)):
            value = self.data[i][column]
            if value not in count_dict:
                count_dict[value] = 1
            else:
                count_dict[value] += 1
        return count_dict

    def separateClassFromData(self):
        result = []
        for i in range(0,len(self.data)):
            result.append([])
            for j in range(0,len(self.data[i])):
                if j is self.class_location:continue
                result[i].append(float(self.data[i][j]))
        return result

    def joinClassAndData(self, data):
        result = []
        for i in range(0,len(data)):
            result.append([])
            for j in range(0,len(data[i])):
                if j is self.class_location:
                    result[i].append(self.data[i][self.class_location])
                result[i].append(data[i][j])
        return result

    def generateNumericMap(self):
        result = False
        for i in range(0,len(self.data[0])):
            if i is self.class_location:
                continue
            try:
                float(self.data[0][i])
                self.continuous_map.append(True)
            except ValueError:
                result = True
                self.continuous_map.append(False)
        return result



    def runAlgorithm(self,algorithm: Algorithm):
        algorithm.run(self)

    # data impution using mean value
    def imputeData(self, missing_value, columns):
        for column_index in columns:
            row_total = 0
            missing_value_indexes = []
            for row_index in self.data:
                if (self.data[row_index[column_index]] is missing_value):
                    missing_value_indexes.append([row_index, column_index])
                else:
                    row_total += self.data[row_index[column_index]]
            if len(missing_value_indexes) is 0:
                print("No data to impute for: " + self.file_path + " with column " + str(column_index))
            # we don't want to count missing values as part of the valid means
            mean = row_total / (len(self.data) - len(missing_value_indexes))
            print("Mean was" + str(mean))

    def makeRandomMap(self, bins):
        temp_data = self.data[0:]
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