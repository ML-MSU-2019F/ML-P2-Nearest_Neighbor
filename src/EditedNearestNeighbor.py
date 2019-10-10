from KNearestNeighbor import KNearestNeighbor

class EditedNearestNeighbor(KNearestNeighbor):
    example_set = []
    def run(self,data_set):
        self.data_set = data_set
        self.data = data_set.data
        self.example_set = self.data
        remove_list = []
        #doing batch removal
        for i in range(0,len(self.data)):
            one = self.example_set[i]
            all = self.getAllButIndex(i)
            nearest = self.getNearestNeighbor(one,all)
            if(self.classify(one[self.data_set.class_location],nearest) is 0):
                remove_list.append(i)
        print(remove_list)

    def getAllButIndex(self,index):
        result = []
        for i in range(0,len(self.data)):
            result.append(i)
            if i is index:
                continue
        return result