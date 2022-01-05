import math
import os
import sys
import time

import cv2
import FormatPhoto
import numpy as np
from matplotlib import pyplot as plt


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
            cv2.imwrite(path_pairs + str(z) + ".jpg", prime)
            z += 1


def background_removal():
    img = cv2.imread('./clothes/pants/IMG_0672.jpg')
    plt.imshow(img), plt.show()
    color = img[0:5, 0:5]
    print(img[0:0, 0:0])
    p_color = img[1000:1005, 1000:1005]
    # print(color, color.shape)
    # print("\nP", p_color, p_color.shape)
    dist_r = np.sum(np.average((color[:, :, 0] - p_color[:, :, 0]), axis=0) ** 2) ** .5
    dist_g = np.sum(np.average((color[:, :, 1] - p_color[:, :, 1]), axis=0) ** 2) ** .5
    dist_b = np.sum(np.average((color[:, :, 2] - p_color[:, :, 2]), axis=0) ** 2) ** .5
    tot = (dist_r + dist_g + dist_b) / 3.
    print(dist_b)
    print(tot, "\n")
    p_color = img[10:15, 10:15]
    dist_r = np.sum(np.average((color[:, :, 0] - p_color[:, :, 0]), axis=0) ** 2) ** .5
    dist_g = np.sum(np.average((color[:, :, 1] - p_color[:, :, 1]), axis=0) ** 2) ** .5
    dist_b = np.sum(np.average((color[:, :, 2] - p_color[:, :, 2]), axis=0) ** 2) ** .5
    tot = (dist_r + dist_g + dist_b) / 3.

    color = img[0:500, 0:img.shape[1]]
    p_color = img[1500:2500, 800:1000]

    # dist_r = np.sum(np.average((color[:, :, 0]), axis=0) ** 2) ** .5
    # dist_g = np.sum(np.average((color[:, :, 1]), axis=0) ** 2) ** .5
    # dist_b = np.sum(np.average((color[:, :, 2]), axis=0) ** 2) ** .5

    p_r = np.average(p_color[:, :, 0])
    p_g = np.average(p_color[:, :, 1])
    p_b = np.average(p_color[:, :, 2])

    color = img[3030:3060, 0:30]
    # print("\n", np.average(color[:, :, 0]))
    # print(np.average(color[:, :, 1]))
    print(img.shape)
    print(int(img.shape[0] / 30))
    print(int(img.shape[1] / 30))



    scalar = 25

    for z in range(3):
        for i in range(int(img.shape[0] / scalar)+1):
            for j in range(int(img.shape[1] / scalar)+1):
                x = j * scalar
                y = i * scalar
                # print(i, j, "\n")
                color = img[y:y + scalar, x:x + scalar]
                try:
                    i_r = np.average(color[:, :, 0])
                    i_g = np.average(color[:, :, 1])
                    i_b = np.average(color[:, :, 2])
                    tot = (p_r - i_r) + (p_g - i_g) + (p_b - i_b)
                    if tot != 0 and np.abs( tot/ 3) > 60:
                        img[y:y + scalar, x:x + scalar] = 255, 255, 255
                except:
                    print("nan")
        scalar -= 5

    plt.imshow(img), plt.show()

    cv2.imwrite("reduced.jpg", img)

    # print(np.average(color, axis=2))
    # print("\n", , axis=2))

    time.sleep(5)


background_removal()
# make_square()
