import os

import cv2
import numpy as np
from matplotlib import pyplot as plt

path_pant = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\pant\\"
path_shirt = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\shirt\\"
path_outfit = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\outfit\\"
path_pair = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\pair\\"
path_shoe = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\shoe\\"
path_coat = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\coat\\"

paths = {"pant": path_pant, "shirt": path_shirt, "outfit": path_outfit, "pair": path_pair, "shoe": path_shoe,
         "coat": path_coat}

images_set = {}


def remove_duplicates(image):
    return image in images_set


def make_square():
    x = 0
    # path = "C:\\Users\\stuar\\Desktop\\TrainingData\\newoutfitclasses"
    path2 = "C:\\Users\\stuar\\Desktop\\TrainingData\\dualclass"
    for folder in os.listdir(path2):
        if folder[0] != "s":
            for filename in os.listdir(path2 + f"/{folder}"):
                p = path2 + "\\" + folder + "\\" + filename
                print(p)
                photo = cv2.imread(p)
                print(path2 + f"\\s{folder}" + filename)
                photo = cv2.resize(photo, (1600, 1600), interpolation=cv2.INTER_AREA)
                plt.imshow(photo), plt.show()
                square = photo[600:1000, 520:1080]
                square = cv2.resize(square, (240, 120), interpolation=cv2.INTER_LINEAR)

                pants = photo[1500:photo.shape[0], 520:700]
                pants = np.concatenate((photo[1500:photo.shape[0], 520:700], pants), axis=0)

                pants = cv2.resize(pants, (square.shape[1], square.shape[0]), interpolation=cv2.INTER_AREA)
                print(pants.shape, square.shape)
                prime = np.concatenate((square, pants), axis=0)
                prime2 = np.concatenate((pants, square), axis=0)
                print(prime.shape)
                plt.imshow(prime), plt.show()
                plt.imshow(prime2), plt.show()
                cv2.imwrite(path2 + f"\\s{folder}\\" + filename, prime2)
                cv2.imwrite(path2 + f"\\s{folder}\\" + "flip_" + filename, prime)
