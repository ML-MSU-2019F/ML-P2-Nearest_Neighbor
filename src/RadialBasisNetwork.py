from MultiLayerPerceptron import MultiLayerPerceptron
from DataSet import DataSet
from NearestNeigbor.KNearestNeighbor import KNearestNeighbor
from DataSet import DataSet
import numpy


class RadialBasisNetwork(MultiLayerPerceptron):

    def __init__(self, data_set: DataSet, clusters, inputs: int, outputs: int,
                 learning_rate):
        self.layers = []
        self.data_set = data_set
        self.clusters = clusters
        self.calculateRadius()
        print("Radius's of clusters calculated")
        self.inputs = inputs
        # one hidden layer
        self.hidden_layers = 1
        # that hidden layer has len(data_set) as nodes
        self.nodes_by_layers = [len(clusters)]
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.regression = False
        print("Constructing Neural Net")
        self.constructInputLayer()
        self.constructHiddenLayers()
        self.constructOutputLayer()
        self.link_layers()
        self.initializeWeights()
        print("Neural Net constructed")

    def calculateRadius(self):
        cluster_radius = []
        headless_clusters = self.data_set.separateClassFromData(data=self.clusters)
        for i in range(0, len(headless_clusters)):
            # k is just going to be 2
            k = 2
            knn = KNearestNeighbor(k, self.data_set)
            # get index, and all but original index, find closest neighbors
            closest = knn.getNearestNeighbor(headless_clusters[i], headless_clusters[:i] + headless_clusters[:i+1])
            cluster_sum = 0
            for j in range(0, len(closest)):
                sub = numpy.subtract(headless_clusters[i], closest[j][0])
                squared = numpy.power(sub, 2)
                cluster_sum += squared
            result_radius = numpy.power(numpy.divide(cluster_sum, k), 0.5)
            cluster_radius.append(result_radius)
        print("")




