import random
import Algorithm
from ReadFile import ReadFile


class DataSet():
    data = None
    file_path = None
    class_location = None
    def __init__(self, file_path, class_location, missing_value=None, columns=None):
        self.class_location = class_location
        self.file_path = file_path
        self.data = ReadFile.read(self,file_path)
        if missing_value is not None and columns is not None:
            self.imputeData(missing_value, columns)

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
        temp_data = self.data
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