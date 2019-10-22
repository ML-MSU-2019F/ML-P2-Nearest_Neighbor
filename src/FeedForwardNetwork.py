from Algorithm import Algorithm
from Node import Node
from Layer import Layer


class FeedForwardNetwork(Algorithm):
    inputs = None
    hidden_layers = None
    nodes_by_layers = []
    layers = []
    outputs = None

    def __init__(self, inputs: int, hidden_layers: int, nodes_by_layers: list, outputs: int):
        self.inputs = inputs
        self.hidden_layers = hidden_layers
        self.nodes_by_layers = nodes_by_layers
        self.outputs = outputs
        self.constructHiddenLayers()

    def inputData(self, input):
        if len(input) is not self.inputs:
            print("Problem occurred, inputs specified: {}, inputs given: {}".format(self.inputs, len(input)))
            exit(1)

    def constructHiddenLayers(self):
        if len(self.nodes_by_layers) != self.hidden_layers:
            print("Problem occurred: Cannot have unspecified hidden nodes")
            print("\tSpecified nodes in layers: {}".format(len(self.nodes_by_layers)))
            print("\tHidden Layers: {}".format(self.hidden_layers))
            exit(1)
        layers = []
        for i in range(0, self.hidden_layers):
            layer = Layer()
            for j in range(0, self.nodes_by_layers[i]):
                layer.addNode(Node(j))
            layers.append(layer)
