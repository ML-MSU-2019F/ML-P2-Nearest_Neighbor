from MLPNode import Node


class Layer:

    def __init__(self, index):
        self.index = index
        self.nodes = []
        self.prev_layer = None
        self.next_layer = None
        self.is_output_layer = False
        self.is_input_layer = False
        self.bias = 1

    def setOutputLayer(self,isOutputLayer):
        self.is_output_layer = isOutputLayer

    def setInputLayer(self, isInputLayer):
        self.is_input_layer = isInputLayer

    def addNode(self, node: Node):
        self.nodes.append(node)

    def setNextLayer(self, layer):
        self.next_layer = layer

    def setPrevLayer(self, layer):
        self.prev_layer = layer
