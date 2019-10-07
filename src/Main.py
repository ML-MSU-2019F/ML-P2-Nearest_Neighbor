from DataSet import DataSet
x_fold_validation = 10
#get data set
abalone = DataSet("../data/abalone.data")
#create a random map to pick from

abalone.makeRandomMap(x_fold_validation)
for i in range(0, x_fold_validation):
    one = abalone.getRandomMap(i)
    rest = abalone.getAllRandomExcept(i)
    #here we would train using rest, then classify/regress one


print("debug")