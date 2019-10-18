from KNearestNeighbor import KNearestNeighbor
from DataSet import DataSet

class EditedNearestNeighbor(KNearestNeighbor):
    data = None
    data_set = None
    def run(self,data_set: DataSet,regression):
        self.data_set = data_set
        #initialize edited set to be the original data
        edited_set = self.data_set.data
        # get original accuracy of ten fold on edited
        self.data_set.makeRandomMap(self.data_set.data,10)
        random_validation_set = self.data_set.getRandomMap(0)
        last_accuracy = self.checkAccuracyAgainstSet(edited_set,random_validation_set)
        print("Accuracy on iteration {} {:2.2f}%".format(0, (last_accuracy) * 100))
        # keep last data in case of validation degredation
        last_data = None
        iterations = 0
        while True:
            # set last data, in case of degredation
            last_data = edited_set[0:]
            # keep a list of elements to remove
            remove_list = []
            # track iterations for logging purposes
            iterations+=1
            # go through our edited set
            for i in range(0,len(edited_set)):
                one = edited_set[i]
                all = self.getAllButIndex(i,edited_set)

                closest = self.getNearestNeighbor(one,all,self.k)
                # if classification of one is wrong, add to remove list
                if(self.classify(one[self.data_set.target_location],closest) is 0):
                    remove_list.append(i)
            remove_offset = 0
            # remove elements from the data set, keeping track of how many we removed so we can remove correctly
            for remove in remove_list:
                print("Deleting index {}".format(remove-remove_offset))
                del edited_set[remove - remove_offset]
                remove_offset+=1
            # check accuracy using a validation set
            accuracy = self.checkAccuracyAgainstSet(edited_set,random_validation_set)
            print("Accuracy on iteration {} {:2.2f}%".format(iterations,(accuracy) * 100))
            # if our accuracy was better, continue
            if accuracy > last_accuracy:
                last_accuracy = accuracy
            # if our accuracy was worse, break
            else:
                break
        # return best accuracy, and store the result of the algorithm
        print("Best Accuracy: {:2.2f}%".format(last_accuracy * 100))
        self.data_set.algo_result = last_data

    # get all but an index within a set, and return it
    def getAllButIndex(self,index,example_set):
        result = []
        for i in range(0,len(example_set)):
            result.append(example_set[i])
            if i is index:
                continue
        return result