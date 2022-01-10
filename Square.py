import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

import Directories


def make_square():
    path = Directories.path_dual
    for folder in os.listdir(path):
        if folder[0] != "s":
            for filename in os.listdir(path + f"/{folder}"):
                p = path + "\\" + folder + "\\" + filename
                print(p)
                photo = cv2.imread(p)
                print(path + f"\\s{folder}" + filename)
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
                plt.imshow(prime), plt.show()
                plt.imshow(prime2), plt.show()
                cv2.imwrite(path + f"\\s{folder}\\" + filename, prime2)
                cv2.imwrite(path + f"\\s{folder}\\" + "flip_" + filename, prime)


# make_square()
