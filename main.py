import os
import time

import cv2
import FormatPhoto
import numpy as np
from matplotlib import pyplot as plt


def make_square():
    path = "C:\\Users\\stuar\\Desktop\\TrainingData\\newoutfitclasses"
    # path2 = "C:\\Users\\stuar\\Desktop\\TrainingData\\squares"
    for folder in os.listdir("C:/Users/stuar/Desktop/TrainingData/newoutfitclasses"):
        for filename in os.listdir(path + f"/{folder}"):
            p = path + "\\" + folder + "\\" + filename
            print(p)
            photo = cv2.imread(p)

            photo = cv2.resize(photo, (1200, 1000), interpolation=cv2.INTER_AREA)
            square = photo[350:700, 400:800]
            pants = photo[900:photo.shape[0], 400:500]
            pants = cv2.resize(pants, (square.shape[1], square.shape[0]), interpolation=cv2.INTER_AREA)
            print(pants.shape, square.shape)
            prime = np.concatenate((square, pants), axis=0)
            prime2 = np.concatenate((pants, square), axis=0)
            plt.imshow(prime), plt.show()
            plt.imshow(prime2), plt.show()
            cv2.imwrite(path2 + "\\s" + folder + "\\" + filename, prime2)
            cv2.imwrite(path2 + "\\s" + folder + "\\" + "flip_" + filename, prime)


def make_pairs():
    x = []
    y = []
    dim = (0, 0)
    path = "C:\\Users\\stuar\\Desktop\\TrainingData\\photosforgen\\p"
    path_shirt = "C:\\Users\\stuar\\Desktop\\TrainingData\\photosforgen\\s"
    path_pairs = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\pairs\\"
    for filename in os.listdir(path):
        p = path + "\\" + filename

        photo = cv2.imread(p)
        photo = cv2.resize(photo, (1200, 1000), interpolation=cv2.INTER_AREA)
        square = photo[200:400, 300:400]
        square = cv2.resize(square, (400, 200), interpolation=cv2.INTER_AREA)
        # pant = FormatPhoto.findAndCut(FormatPhoto.noiseReduction(photo, 10), "pant")
        dim = square.shape
        print(dim)
        x.append(square)


        # color = FormatPhoto.getColor(photo)


    for shirts in os.listdir(path_shirt):
        shirt = cv2.imread(path_shirt + "\\" + shirts)
        shirt = cv2.resize(shirt, (1200, 1000), interpolation=cv2.INTER_AREA)
        shirt = FormatPhoto.findAndCut(FormatPhoto.noiseReduction(shirt, 10), "shirt")
        shirt = shirt[300:650, 400: 800]
        shirt = cv2.resize(shirt, (dim[1], dim[0]), interpolation=cv2.INTER_AREA)
        y.append(shirt)
        print(y)

    z = 1
    for xs in x:
        for ys in y:
            prime = np.concatenate((ys, xs), axis=0)
            plt.imshow(prime), plt.show()
            cv2.imwrite(path_pairs+ str(z) +".jpg", prime)
            z += 1


make_square()
