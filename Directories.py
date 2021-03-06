import pymongo

path_pant = "C:\\Users\\stuar\\Desktop\\OGenerator\\clothes\\pant\\"
path_shirt = "C:\\Users\\stuar\\Desktop\\OGenerator\\clothes\\shirt\\"
path_outfit = "C:\\Users\\stuar\\Desktop\\OGenerator\\clothes\\outfit\\"
path_pair = "C:\\Users\\stuar\\Desktop\\OGenerator\\clothes\\pair\\"
path_shoe = "C:\\Users\\stuar\\Desktop\\OGenerator\\clothes\\shoe\\"
path_coat = "C:\\Users\\stuar\\Desktop\\OGenerator\\clothes\\coat\\"
path_dual = "C:\\Users\\stuar\\Desktop\\TrainingData\\dualclass"
dirtrain = "/Users/stuar/Desktop/TrainingData/dualclass/train"
dirtest = "/Users/stuar/Desktop/TrainingData/dualclass/test"
dir_large = "/Users/stuar/Desktop/TrainingData/unsupervised"
dir_one = "/Users/stuar/Desktop/TrainingData/unsup"
dir_pred = "clothes/pair"

dir_1 = "./clothes/predicted_1"
dir_15 = "./clothes/predicted_1dot5"
dir_5 = "./clothes/predicted_dot5"
dir_zip = "./clothes/predicted_zip"

clothing_folders = {"pant": path_pant, "shirt": path_shirt, "outfit": path_outfit, "pair": path_pair, "shoe": path_shoe,
                    "coat": path_coat}
fashionable_output = {1: dir_1, 1.5: dir_15, .5: dir_5, 0: dir_zip}

neural_net = {"train": dirtrain, "test": dirtest, "outfit": dir_pred}

myclient = pymongo.MongoClient("mongodb://localhost:27017/")
databases = myclient.list_database_names()
mydb = myclient["fashionphotos"]

collection_types = {"outfit": mydb.outfit, "shoe": mydb.shoes, "coat": mydb.coat, "shirt": mydb.shirt,
                    "pant": mydb.pant, "short": mydb.short, "pair": mydb.pair, "stack": mydb.stack}
