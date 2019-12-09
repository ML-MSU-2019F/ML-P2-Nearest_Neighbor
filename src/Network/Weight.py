

class Weight:
    """
    Dynamically hold weights as an object reference, this enables quick reference of weight structure
    in an MLP than can be modified at a top level where changes propagate down
    """
    def __init__(self):
        self.weight = None

    def setWeight(self, weight):
        self.weight = weight