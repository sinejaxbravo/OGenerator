path_pant = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\pant\\"
path_shirt = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\shirt\\"
path_outfit = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\outfit\\"
path_pair = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\pair\\"
path_shoe = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\shoe\\"
path_coat = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\coat\\"
path2 = "C:\\Users\\stuar\\Desktop\\TrainingData\\dualclass"
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
         "coat": path_coat }
fashionable_output = {1: dir_1, 1.5: dir_15, .5: dir_5, 0: dir_zip}

neural_net = {"train": dirtrain, "test": dirtest, "outfit": dir_pred}
