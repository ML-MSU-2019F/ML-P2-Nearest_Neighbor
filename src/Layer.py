from MLPNode import Node


class Layer:
    """
    Basic class that holds nodes and local variables relating to the nodes within it.
    """
    def __init__(self, index):
        # defining instance based variables
        self.index = index
        self.nodes = []
        self.prev_layer = None
        self.next_layer = None
        self.is_output_layer = False
        self.is_input_layer = False

    # set if this layer is an output layer
    def setOutputLayer(self,isOutputLayer):
        self.is_output_layer = isOutputLayer

    # set if this layer is an input layer
    def setInputLayer(self, isInputLayer):
        self.is_input_layer = isInputLayer

    # add node to list of nodes in layer
    def addNode(self, node: Node):
        self.nodes.append(node)

    # set the reference to the next layer, used for linking purposes
    def setNextLayer(self, layer):
        self.next_layer = layer

    # set the reference for the previous layer, used for linking purposes
    def setPrevLayer(self, layer):
        self.prev_layer = layer
