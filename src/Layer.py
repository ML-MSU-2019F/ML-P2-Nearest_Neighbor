from Node import Node


class Layer:
    prev_layer = None
    next_layer = None
    nodes = []

    def __init__(self):
        pass

    def addNode(self, node: Node):
        self.nodes.append(node)

    def setNextLayer(self, layer):
        self.next_layer = layer

    def setPrevLayer(self, layer):
        self.prev_layer = layer
