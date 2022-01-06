import math
import os
import sys
import time

import cv2
import skimage

import FormatPhoto
import numpy as np
from matplotlib import pyplot as plt

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
from DB import DB

path_pant = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\pant\\"
path_shirt = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\shirt\\"
path_outfit = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\outfit\\"
path_pair = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\pair\\"
path_shoe = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\shoe\\"
path_coat = "C:\\Users\\stuar\\Desktop\\PyProjects\\clothes\\coat\\"

paths = {"pant": path_pant, "shirt": path_shirt, "outfit": path_outfit, "pair": path_pair, "shoe": path_shoe,
         "coat": path_coat}


# TODO REDUCED IS THE BACKGROUND REMOVED. ITEM IS JUST THE SQUARE
def clean_item(path_and_name, clothing_type, dim=(400, 200)):
    item = cv2.imread(path_and_name)
    item = cv2.resize(item, (1200, 1000), interpolation=cv2.INTER_AREA)
    reduced = FormatPhoto.findAndCut(FormatPhoto.noiseReduction(item, 10), clothing_type)
    if type == "pant":
        item = reduced[200:400, 300:400]
        item = cv2.resize(item, (dim[0], dim[1]), interpolation=cv2.INTER_AREA)
    else:
        item = reduced[300:650, 400:800]
        item = cv2.resize(item, (dim[0], dim[1]), interpolation=cv2.INTER_AREA)

    return reduced, item


# TODO save the square of each photo with a link to it (and its name)so in the future we can simply pair
#  the new item with all of the and dont have to iterate through and recut all of the previous squares

def make_pairs(paths_list, paths_names, output_path, dim=(400, 200)):
    all_irrespective_of_types = []
    items_verified = []
    index = 0
    for path in paths_list:
        path_items = []
        for item_name in os.listdir(path):
            path_and_name = path + "\\" + item_name
            # item = cv2.imread(path + "\\" + item_name)
            # item = cv2.resize(item, (1200, 1000), interpolation=cv2.INTER_AREA)
            # item = FormatPhoto.findAndCut(FormatPhoto.noiseReduction(item, 10), f"{paths_names[index]}")
            # print(paths_names[index])
            if paths_names[index] == "pant":
                reduced, item = clean_item(path_and_name, "pant", dim)
                # item = item[200:400, 300:400]
                # item = cv2.resize(item, (dim[0], dim[1]), interpolation=cv2.INTER_AREA)
            else:
                reduced, item = clean_item(path_and_name, "else", dim)
                # item = item[300:650, 400:800]
                # item = cv2.resize(item, (dim[0], dim[1]), interpolation=cv2.INTER_AREA)
            obj = (path_and_name, item)
            path_items.append(obj)
            all_irrespective_of_types.append(obj)
            plt.imshow(item), plt.show()
            items_verified.append(path_items)
        index += 1

        # color = FormatPhoto.getColor(photo)

    db = DB()
    combo_num = db.collection_types["pair"].count_documents({})
    paired = []
    for lists in items_verified:
        for list_items in lists:
            for item in all_irrespective_of_types:
                set_pair = {item[0], list_items[0]}
                if item not in lists and set_pair not in paired:
                    comb = np.concatenate((item[1], list_items[1]), axis=0)
                    plt.imshow(comb), plt.show()
                    combo = f"{item[0]}, {list_items[0]}"
                    name = f"pair_{combo_num}.jpg"
                    print(output_path)
                    print(name)
                    print(combo)
                    paired.append(set_pair)
                    ret = db.store_image(comb, output_path, f"pair_{combo_num}.jpg", "pair", combo)
                    print(ret)
                    # image = {
                    #     "name": name,
                    #     "location": location,
                    #     "date": datetime.datetime.utcnow(),
                    #     "combination": combination
                    # }
                    # cv2.imwrite(path_pairs + str(z) + ".jpg", prime)
                    combo_num += 1


def set_up_singular_item(list_of_to_be_done):
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
