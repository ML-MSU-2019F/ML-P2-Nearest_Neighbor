from EditedNearestNeighbor import EditedNearestNeighbor
from KNearestNeighbor import KNearestNeighbor
import numpy
class KMeans(KNearestNeighbor):
    def __init__(self, enn_result, k_centers):
        data = numpy.array(enn_result)
        centroids = []
        for i in range(0,k_centers):
            rand = numpy.random.randint(0,len(data),1)
            centroids.append(data[rand])
            del data[rand]

        for line in data:
            one = line
            all = centroids
            # get nearest centroid
            closest = self.getNearestNeighbor(one,all,1)



