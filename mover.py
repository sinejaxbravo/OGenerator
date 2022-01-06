import shutil
import os




# shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
sets = {"1", "2"}
def make_folder(mode, f):
    for x in f:
        s = x[0: x.index("/")]
        if mode == "train":
            folder = dirtrain + s
        elif mode == "test":
            folder = dirtest + s
        else:
            return
        os.mkdir(folder)


def move_files(mode, f):
    for x in f:
        if mode == "train":
            s = dirtrain + x + ".jpg"
        elif mode == "test":
            s = dirtest + x + ".jpg"
        else:
            return
        file = trainpath + x + ".jpg"
        shutil.move(file.replace("\n", ""), s.replace("\n", ""))


# Make sure you have set up the test and the train folders
# Make the directories before you move anything!

# photos = open("../Udemy/Udemy/test.txt", "rt")
# dirtrain = "/Users/stuar/Desktop/TrainingData/food-101/train/"
# dirtest = "/Users/stuar/Desktop/TrainingData/food-101/test/"
# trainpath = "/Users/stuar/Desktop/TrainingData/food-101/food-101/images/"