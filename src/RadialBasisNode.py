

class RadialBasisNode:
    def __init__(self, index, learning_rate=0.1, momentum_constant=.5):
        self.index = index
        self.error = None
        self.learning_rate = learning_rate
        self.momentum_constant = momentum_constant;
        self.distance = None
        self.momentum = None
        self.derived_times_errors = []
        self.weights = []
        self.layer = None
        self.output = None
        self.override_input = None
        self.network = None