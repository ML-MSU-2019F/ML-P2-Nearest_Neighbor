from KNearestNeighbor import KNearestNeighbor
import numpy
class EditedNearestNeighbor(KNearestNeighbor):
    example_set = []
    data = None
    data_set = None
    final_data = None
    def run(self,data_set):
        self.data_set = data_set
        edited_set = self.data_set.data

        original_result = self.runTenFold(edited_set)
        last_accuracy = original_result[1] / original_result[0]
        print("Accuracy on iteration {} {:2.2f}".format(0, (last_accuracy) * 100))

        #doing batch removal
        last_data = None
        iterations = 0
        while True:
            last_data = edited_set[0:]
            remove_list = []
            iterations+=1
            for i in range(0,len(edited_set)):
                one = edited_set[i]
                all = self.getAllButIndex(i,edited_set)
                closest = self.getNearestNeighbor(one,all)
                if(self.classify(one[self.data_set.class_location],closest) is 0):
                    remove_list.append(i)
            remove_offset = 0
            for remove in remove_list:
                del edited_set[remove - remove_offset]
                remove_offset+=1
            result = self.runTenFold(edited_set)
            accuracy = result[1] / result[0]
            print("Accuracy on iteration {} {:2.2f}".format(iterations,(accuracy) * 100))
            if accuracy > last_accuracy:
                last_accuracy = accuracy
            else:
                break
        self.final_data = last_data
    def getAllButIndex(self,index,example_set):
        result = []
        for i in range(0,len(example_set)):
            result.append(example_set[i])
            if i is index:
                continue
        return result