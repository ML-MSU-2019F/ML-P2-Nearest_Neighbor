from KNearestNeighbor import KNearestNeighbor
import numpy
class CondensedNearestNeighbor(KNearestNeighbor):
    example_set = []
    data = None
    data_set = None
    def run(self,data_set):
        self.data_set = data_set
        original_data = self.data_set.data[0:]
        # condensed set
        self.data_set.data = []
        last_accuracy = 0
        print("Accuracy on iteration {} {:2.2f}".format(0, (last_accuracy) * 100))
        # doing batch removal
        last_data = None
        iterations = 0
        original_point = original_data.pop()
        self.data_set.data.append(original_point)
        degraded = False
        done = False
        while not degraded:
            iterations+=1
            add_list = []
            index = 0
            while True:
                if len(original_data) is 0:
                    # is empty
                    done = True
                    break
                one = original_data[index]
                all = self.data_set.data
                closest = self.getNearestNeighbor(one,all,1)
                if(self.classify(one[self.data_set.class_location],closest) is 0):
                    add_list.append(index)
                else:
                    pass
                    # print("Classified Right!")
                index += 1
                if index >= len(original_data):
                    if len(add_list) is 0:
                        done = True
                    break
                if (len(add_list) == 50):
                    break
            add_offset = 0
            for i in add_list:
                self.data_set.data.append(original_data[i - add_offset])
                del original_data[i-add_offset]
                add_offset += 1
            index += 1
            result = self.runTenFold()
            accuracy = result[1] / result[0]
            print(result[1])
            print(result[0])
            print("Accuracy on iteration {} {:2.2f}".format(iterations, (accuracy) * 100))
            if done:
                break





    def getAllButIndex(self,index,example_set):
        result = []
        for i in range(0,len(example_set)):
            result.append(example_set[i])
            if i is index:
                continue
        return result