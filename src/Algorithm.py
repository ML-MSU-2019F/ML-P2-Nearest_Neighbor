from abc import ABC, abstractmethod


class Algorithm(ABC):
    """
    Algorithm class allows for predictable behavior for machine learning algorithms
    """
    def __init__(self):
        self.data_set = None

    def run(self, DataSet):
        pass

    def checkAccuracy(self):
        pass
