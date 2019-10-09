from abc import ABC, abstractmethod
class Algorithm(ABC):
    data_set = None
    def run(self,DataSet):
        pass
    def checkAccuracy(self):
        pass
