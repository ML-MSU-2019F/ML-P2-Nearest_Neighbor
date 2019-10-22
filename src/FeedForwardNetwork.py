from Algorithm import Algorithm
from Node import Node
from Layer import Layer
from DataSet import DataSet


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
        print("Constructing Neural Net")
        self.constructInputLayer()
        self.constructHiddenLayers()
        self.constructOutputLayer()
        self.link_layers()
        self.initializeWeights()
        print("Neural Net constructed")

    # set to false for now, but will change later
    def run(self, data_set: DataSet, regression=False):
        data = data_set.separateClassFromData()
        # input values into the input layer
        for line in data:
            if len(line) != self.inputs:
                print("Error, we need to have as many inputs as features")
                print("We have {} features, but only {} layers".format(len(line), self.inputs))
                exit(1)
            for i in range(0, len(line)):
                # input values into first layer
                self.layers[0].nodes[i].overrideInput(line[i])
            # go though each layer, running based on set input
            for i in range(0, len(self.layers)):
                for j in range(0, len(self.layers[i].nodes)):
                    self.layers[i].nodes[j].run()


    def link_layers(self):
        for i in range(0,len(self.layers)):
            # is not last layer
            if i+1 is not len(self.layers):
                self.layers[i].next_layer = self.layers[i+1]
            # is not first layer
            if i-1 is not -1:
                self.layers[i].prev_layer = self.layers[i-1]

    def inputData(self, input):
        if len(input) is not self.inputs:
            print("Problem occurred, inputs specified: {}, inputs given: {}".format(self.inputs, len(input)))
            exit(1)

    def initializeWeights(self):
        # iterate through all but last layer
        for i in range(0, len(self.layers)):
            # initialize weights on all nodes
            for j in range(0, len(self.layers[i].nodes)):
                self.layers[i].nodes[j].initWeights(-0.3, 0.3)

    def constructInputLayer(self):
        layer = Layer(len(self.layers))
        for i in range(0, self.inputs):
            node_i = Node(i)
            node_i.setLayer(layer)
            layer.addNode(node_i)
        layer.setInputLayer(True)
        self.layers.append(layer)


    def constructOutputLayer(self):
        layer = Layer(len(self.layers))
        for i in range(0, self.outputs):
            node_i = Node(i)
            node_i.setLayer(layer)
            layer.addNode(node_i)
        layer.setOutputLayer(True)
        self.layers.append(layer)

    def constructHiddenLayers(self):
        if len(self.nodes_by_layers) != self.hidden_layers:
            print("Problem occurred: Cannot have unspecified hidden nodes")
            print("\tSpecified nodes in layers: {}".format(len(self.nodes_by_layers)))
            print("\tHidden Layers: {}".format(self.hidden_layers))
            exit(1)

        for i in range(0, self.hidden_layers):
            layer = Layer(len(self.layers))
            for j in range(0, self.nodes_by_layers[i]):
                node_j = Node(j)
                node_j.setLayer(layer)
                layer.addNode(node_j)
            self.layers.append(layer)
