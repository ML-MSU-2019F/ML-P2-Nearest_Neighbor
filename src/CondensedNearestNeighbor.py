from KNearestNeighbor import KNearestNeighbor
import numpy
class CondensedNearestNeighbor(KNearestNeighbor):
    example_set = []
    data = None
    data_set = None
    final_data = None
    def run(self,data_set,regression):
        self.data_set = data_set
        original_data = self.data_set.data[0:]
        condensed_set = []
        # condensed set
        last_accuracy = 0
        print("Accuracy on iteration {} {:2.2f}".format(0, (last_accuracy) * 100))
        # doing batch removal
        last_data = None
        iterations = 0
        original_point = original_data.pop()
        condensed_set.append(original_point)
        degraded = False
        done = False
        while not degraded:
            last_data = condensed_set[0:]
            iterations+=1
            add_list = []
            index = 0
            while True:
                if len(original_data) is 0:
                    # is empty
                    done = True
                    break
                one = original_data[index]
                all = condensed_set
                closest = self.getNearestNeighbor(one,all,1)
                if(self.classify(one[self.data_set.target_location],closest) is 0):
                    add_list.append(index)
                index += 1
                if index >= len(original_data):
                    if len(add_list) is 0:
                        done = True
                    break
                if (len(add_list) == 50):
                    break
            add_offset = 0
            for i in add_list:
                condensed_set.append(original_data[i - add_offset])
                del original_data[i-add_offset]
                add_offset += 1
            index += 1
            result = self.runTenFold(condensed_set)
            accuracy = result[1] / result[0]
            print(result[1])
            print(result[0])
            print("Accuracy on iteration {} {:2.2f}".format(iterations, (accuracy) * 100))
            if done:
                break
        self.final_data = last_data
