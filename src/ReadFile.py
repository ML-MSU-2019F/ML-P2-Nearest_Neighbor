import os


class ReadFile:
    """
    Provide basic reading functionality from disk, returns data as a two dimensional array
    """
    staticmethod
    def read(self,file_path):
        data_array = []
        if os.path.exists(file_path) == 0:
            print("File " + file_path + " did not exist")
            return;

        file = open(file_path, "r+")
        for line in file:
            line = line.rstrip('\n');
            data_array.append(line.split(","))
        return data_array