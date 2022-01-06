import math
import os
import sys
import time

import cv2
import skimage

import FormatPhoto
import numpy as np
from matplotlib import pyplot as plt

from DB import DB

path_pant = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\pant\\"
path_shirt = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\shirt\\"
path_outfit = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\outfit\\"
path_pair = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\pair\\"
path_shoe = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\shoe\\"
path_coat = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\coat\\"

paths = {"pant": path_pant, "shirt": path_shirt, "outfit": path_outfit, "pair": path_pair, "shoe": path_shoe,
         "coat": path_coat}


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


def make_pairs(paths, paths_names, output_path, dim=(400, 200)):
    all = []
    items_verified = []
    index = 0
    for path in paths:
        path_items = []
        for item_name in os.listdir(path):
            item = cv2.imread(path + "\\" + item_name)
            item = cv2.resize(item, (1200, 1000), interpolation=cv2.INTER_AREA)
            item = FormatPhoto.findAndCut(FormatPhoto.noiseReduction(item, 10), f"{paths_names[index]}")
            item = item[200:400, 300:400]
            item = cv2.resize(item, (dim[1], dim[0]), interpolation=cv2.INTER_AREA)
            dim = item.shape
            path_items.append((item_name, item))
            all.append((item_name, item))
            plt.imshow(item), plt.show()
        items_verified.append(path_items)

        # color = FormatPhoto.getColor(photo)

    z = 0
    ind_x = 0
    ind_y = 0
    db = DB()
    z = db.collection_types["pair"].count_documents({})
    combos = []

    for lists in items_verified:
        for list_items in lists:
            for item in all:
                if item not in list_items:
                    comb = np.concatenate((item[1], list_items[1]), axis=0)
                    plt.imshow(comb), plt.show()
                    combo = f"{item[0]}, {list_items[0]}"
                    name = f"pair_{z}.jpg"
                    print(output_path)
                    print(name)
                    print(combo)
                    # ret = db.store_image(prime, path_pairs, f"pair_{z}.jpg", "outfit", combo)
                    # print(ret)
                    # image = {
                    #     "name": name,
                    #     "location": location,
                    #     "date": datetime.datetime.utcnow(),
                    #     "combination": combination
                    # }
                    # cv2.imwrite(path_pairs + str(z) + ".jpg", prime)
                    z += 1


def set_up_singular(list_of_to_be_done):
    # def store_image(self, image, location, name, type_of, combination=None):

    for f in list_of_to_be_done:
        item = cv2.imread(f)


def background_removal():
    img = cv2.imread('./clothes/shirts/IMG_0671.jpg')
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
        for i in range(int(img.shape[0] / scalar) + 1):
            for j in range(int(img.shape[1] / scalar) + 1):
                x = j * scalar
                y = i * scalar
                # print(i, j, "\n")
                color = img[y:y + scalar, x:x + scalar]
                try:
                    i_r = np.average(color[:, :, 0])
                    i_g = np.average(color[:, :, 1])
                    i_b = np.average(color[:, :, 2])
                    tot = (p_r - i_r) + (p_g - i_g) + (p_b - i_b)
                    if tot != 0 and np.abs(tot / 3) > 60:
                        img[y:y + scalar, x:x + scalar] = 255, 255, 255
                except:
                    print("nan")
        scalar -= 5

    plt.imshow(img), plt.show()
    cv2.imwrite('reduced.jpg', img)


# background_removal()
# make_square()

paths_to_use = [paths["pant"], paths["shirt"], paths["coat"]]
names = ["pant", "shirt", "coat"]
output_path_to_use = paths["pair"]
make_pairs(paths_to_use, names, output_path_to_use)
