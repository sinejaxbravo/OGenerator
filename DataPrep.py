
import os
import time
import cv2
import mediapipe as mp
import FormatPhoto
import numpy as np
from matplotlib import pyplot as plt
from DB import DB

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


def test_removal(image):
    # TODO MAKE A SEGMENTATION NN TO REMOVE BACKGROUND
    image = cv2.imread("./clothes/shirt/IMG_0675.jpg")
    RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mp_selfie_segmentation = mp.solutions.selfie_segmentation
    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)
    results = selfie_segmentation.process(RGB)
    mask = results.segmentation_mask
    cv2.imshow("mask", cv2.resize(mask, (300, 300), interpolation=cv2.INTER_AREA))
    cv2.imshow("image", cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA))
    time.sleep(10)



# TODO REDUCED IS THE BACKGROUND REMOVED. ITEM IS JUST THE SQUARE
def clean_item(path_and_name, clothing_type, dim=(400, 200), iterations=5):
    item = cv2.imread(path_and_name)
    item = cv2.resize(item, (1200, 1000), interpolation=cv2.INTER_AREA)
    if clothing_type != "shoe":
        reduced = FormatPhoto.findAndCut(FormatPhoto.noiseReduction(item, iterations), clothing_type)
    else:
        reduced = item
    if clothing_type == "pant":
        item = reduced[200:400, 300:400]
        item = cv2.resize(item, (dim[0], dim[1]), interpolation=cv2.INTER_AREA)
    elif clothing_type == "shoe":
        sq = item[300:1000, 300:900]
        item = np.full((sq.shape[0], sq.shape[1], 3), sq)
        item = cv2.resize(item, (dim[0], dim[1]), interpolation=cv2.INTER_AREA)
    else:
        item = reduced[200:500, 400:800]
        item = cv2.resize(item, (dim[0], dim[1]), interpolation=cv2.INTER_AREA)
    plt.imshow(item), plt.show()
    return reduced, item


def make_squares(path, item_type, dim=(400, 200)):
    path_items = []
    for item_name in os.listdir(path):
        path_and_name = path + item_name
        if item_type == "pant":
            reduced, item = clean_item(path_and_name, "pant", dim)
        elif item_type == "shoe":
            reduced, item = clean_item(path_and_name, "shoe", dim, iterations=0)
        else:
            reduced, item = clean_item(path_and_name, "else", dim)
        obj = (path_and_name, item)
        path_items.append(obj)
        plt.imshow(item), plt.show()
    return path_items



# TODO save the square of each photo with a link to it (and its name)so in the future we can simply pair
#  the new item with all of the and dont have to iterate through and recut all of the previous squares

def make_pairs(paths_list, paths_names, output_path, dim=(400, 200), with_squares=True, images=[], all_=[]):
    all_irrespective_of_types = []
    items_verified = []
    index = 0
    if with_squares:
        for path in paths_list:
            path_items = []
            for item_name in os.listdir(path):
                path_and_name = path + "\\" + item_name
                if paths_names[index] == "pant":
                    reduced, item = clean_item(path_and_name, "pant", dim)
                else:
                    reduced, item = clean_item(path_and_name, "else", dim)
                obj = (path_and_name, item)
                path_items.append(obj)
                all_irrespective_of_types.append(obj)
                plt.imshow(item), plt.show()
                items_verified.append(path_items)
            index += 1
    else:
        items_verified = images
        all_irrespective_of_types = all_


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
                    paired.append(set_pair)
                    ret = db.store_image(comb, output_path, f"pair_{combo_num}.jpg", "pair", combo)
                    print(ret)
                    combo_num += 1
    return paired

# TODO THIS METHOD THAT TAKES A SQUARE AND CREATES ALL NECESSARY COMBINATIONS
def set_up_singular_item(list_of_to_be_done):

    for f in list_of_to_be_done:
        item = cv2.imread(f)


# TODO IGNORE
def background_removal():
    img = cv2.imread('./clothes/shirts/IMG_0671.jpg')
    plt.imshow(img), plt.show()
    color = img[0:5, 0:5]
    p_color = img[1000:1005, 1000:1005]
    dist_r = np.sum(np.average((color[:, :, 0] - p_color[:, :, 0]), axis=0) ** 2) ** .5
    dist_g = np.sum(np.average((color[:, :, 1] - p_color[:, :, 1]), axis=0) ** 2) ** .5
    dist_b = np.sum(np.average((color[:, :, 2] - p_color[:, :, 2]), axis=0) ** 2) ** .5
    tot = (dist_r + dist_g + dist_b) / 3.
    p_color = img[10:15, 10:15]
    dist_r = np.sum(np.average((color[:, :, 0] - p_color[:, :, 0]), axis=0) ** 2) ** .5
    dist_g = np.sum(np.average((color[:, :, 1] - p_color[:, :, 1]), axis=0) ** 2) ** .5
    dist_b = np.sum(np.average((color[:, :, 2] - p_color[:, :, 2]), axis=0) ** 2) ** .5
    tot = (dist_r + dist_g + dist_b) / 3.

    color = img[0:500, 0:img.shape[1]]
    p_color = img[1500:2500, 800:1000]


    p_r = np.average(p_color[:, :, 0])
    p_g = np.average(p_color[:, :, 1])
    p_b = np.average(p_color[:, :, 2])

    color = img[3030:3060, 0:30]
    scalar = 25

    for z in range(3):
        for i in range(int(img.shape[0] / scalar) + 1):
            for j in range(int(img.shape[1] / scalar) + 1):
                x = j * scalar
                y = i * scalar

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


# make_squares(Directories.clothing_folders["shoe"], "shoe")


# background_removal()
# make_square()

# paths_to_use = [paths["pant"], paths["shirt"], paths["coat"]]
# names = ["pant", "shirt", "coat"]
# output_path_to_use = paths["pair"]
# make_pairs(paths_to_use, names, output_path_to_use)
