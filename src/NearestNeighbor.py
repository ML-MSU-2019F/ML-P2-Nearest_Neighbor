from abc import ABC, abstractmethod
from Algorithm import Algorithm
class NearestNeighbor(Algorithm):
    classify_or_regress = None
    #constructor
    def __init__(self,classify_or_regress):
        self.classify_or_regress = classify_or_regress
        pass
    def getNearestNeighbor(self,data_line):
        pass
    def classify(self):
        pass
    def regress(self):
        pass



