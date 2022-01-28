import asyncio
import math
import os
import random
import time

import cv2
import numpy as np
import scipy.stats
from keras.preprocessing import image
from matplotlib import pyplot as plt

from scipy.spatial import distance
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler
import progressbar

import Directories
import UnsupervisedClustering
from Weather import extract_weather
import Directories as dir
from DB import DB
from DataPrep import make_pairs, make_squares

sesh = asyncio.new_event_loop()
temperature, precipitation, overcast = sesh.run_until_complete(extract_weather())
# print(temperature)
# print(temperature)

paths = dir.clothing_folders

paths_to_use = [paths["pant"], paths["shirt"], paths["coat"], paths["shoe"]]
names = ["pant", "shirt", "coat", "shoe"]
output_path_to_use = paths["pair"]
#
#
db = DB()


def stitch(o, t, combo_num, output_path):
    comb = np.concatenate((o[1], t[1]), axis=0)
    plt.imshow(comb), plt.show()
    combo = f"{o[0]}, {t[0]}"
    name = f"pair_{combo_num}.jpg"
    print(output_path)
    print(name)
    print(combo)
    ret = db.store_image(comb, output_path, f"pair_{combo_num}.jpg", "pair", combo)
    print(ret)
    return f"pair_{combo_num}.jpg", comb


def generate_all_from_scratch():
    pants = make_squares(paths["pant"], "pant")
    # TODO UNCOMMENT IF YOU WANT SHOES
    # shoes = make_squares(paths["shoe"], "shoe")

    shirt = make_squares(paths["shirt"], "shirt")

    coat = make_squares(paths["coat"], "coat")

    c_num = 1

    outfit_num = 0
    for p in pants:
        # TODO UNCOMMENT IF YOU WANT SHOES
        # for s in shoes:
        #     retps, ps = stitch(p, s, c_num, output_path_to_use)
        for sh in shirt:
            retpsh, psh = stitch(p, sh, c_num + 1, output_path_to_use)
            woc = [retpsh]
            db.add_outfits(outfit_num, [p[0], sh[0]], "0", 0, woc, "no")
            for c in coat:
                wc = [retpsh]
                ret, pc = stitch(p, c, c_num + 2, output_path_to_use)
                wc.append(ret)
                ret, sc = stitch(sh, c, c_num + 3, output_path_to_use)
                wc.append(ret)
                c_num += 6
                db.add_outfits(outfit_num + 1, [p[0], sh[0], c[0]], "0", 0, wc, "yes")
                outfit_num += 1
            outfit_num += 1


def predict(model, photos):
    for file in photos:
        imag = image.load_img(f"{file}", target_size=(224, 224))
        img_array = image.img_to_array(imag)
        img_batch = np.expand_dims(img_array, 0)
        prediction = model.predict(img_batch)
        print(prediction)


def get_dist(arr):
    tot = 0
    dist = 0.
    comp = set()
    if len(arr) == 1:
        return dist
    for i in arr:
        for j in arr:
            a = (i[0], i[1])
            b = (j[0], j[1])
            to_a = (a, b)
            if to_a not in comp and a != b:
                tot += 1
                dist += math.dist(a, b)
                comp.add(to_a)
                t = (b, a)
                comp.add(t)
    if dist != 0.:
        dist /= tot

    return dist

# TODO NOT USED IN THIS CLASS!
def Affinity(data, labels):
    X = StandardScaler().fit_transform(data)
    model = AffinityPropagation(damping=0.9, random_state=None)
    model.fit(X)
    yhat = model.predict(X)
    clusters = np.unique(yhat)
    return X, labels


def features_lists(model):
    data = {}
    i = 0
    with progressbar.ProgressBar(max_value=len(os.listdir("clothes/pair"))) as bar:
        for filename in os.listdir("clothes/pair"):
            imag = image.load_img(f"clothes/pair/{filename}", target_size=(224, 224))
            bar.update(i)
            img_array = image.img_to_array(imag)
            img_batch = np.expand_dims(img_array, 0)
            extract = UnsupervisedClustering.extraction(model, img_batch)
            data[filename] = extract
            i += 1
        names = np.array(list(data.keys()))
        feats = np.array(list(data.values()))
        feats = feats.reshape((feats.shape[0], feats.shape[2]))
        return names, feats


def update_accuracy():
    res = UnsupervisedClustering.train()
    database = DB()
    outfit = database.collection_types["outfit"]

    pairs, pair_feats = features_lists(res)
    # TODO UNCOMMENT TO USE AFFINITY CLUSTERING
    # X, labels = Affinity(pair_feats, pairs)
    X = StandardScaler().fit_transform(pair_feats)
    print(db.collection_types)
    outfits = outfit.find()
    l_label = pairs.tolist()

    for fit in outfits:
        print("name: ", fit["name"])
        temp = fit["pairs"]
        val = []
        for t in temp:
            print(t)
            ind = l_label.index(t)
            val.append([X[ind, 0], X[ind, 1]])
            print(X[ind, 0], X[ind, 1])

        euc_distance = get_dist(val)

        new_percent = []
        for p in val:
            new_percent.append([-1 * p[0], p[1]])
        print(f"Percent chance:\n {scipy.stats.norm.cdf(new_percent)}", "\n")
        new_percent = np.mean(scipy.stats.norm.cdf(new_percent))
        # fit["fashionable_likelihood"] = new_percent
        outfit.update_one(fit, {"$set": {"fashionable_likelihood": new_percent, "distance": euc_distance}})
        # outfit.insert_one(fit, {"$set", {"distance": euc_distance}})
        print(new_percent, " likelihood saved!")
        print(euc_distance, " distance saved!", "\n\n")
        time.sleep(.01)


def get_best(criteria, sort=True, collection_name=None):
    database = DB()
    tot = 0
    outfit = database.collection_types["outfit"]
    if sort:
        outfits = outfit.find().sort("fashionable_likelihood", 1)
    else:
        outfits = outfit.find(criteria)

    for fit in outfits:
        print("name: ", fit["name"], " distance: ", fit["distance"], " likely: ", fit["fashionable_likelihood"],
              "pairs: ", fit["pairs"])
        ret_code = database.collection_types[collection_name].insert_one(fit).inserted_id
        print(ret_code)
        tot += 1
        time.sleep(.002)
    print(f"\nTotal items: {tot}\n\n\n")


def get_outfit():
    count = db.collection_types["stack"].count_documents({})
    rand = db.collection_types["stack"].find().limit(1).skip(random.randint(0, count))
    session = asyncio.new_event_loop()
    temp, precip, overc = seshion.run_until_complete(extract_weather())
    resp = "y"
    for r in rand:
        if int(temp) < 55 and r["coat"] == "yes":
            print("\nOUTFIT-- name: ", r["name"], " distance: ", r["distance"], " likely: ",
                  r["fashionable_likelihood"],
                  "pairs: ", r["pairs"])
            items = []
            item_str = (r["combination"].replace("\\", "/")).replace(" ", "")
            print(item_str)
            while item_str.find(",") != -1:
                ind = item_str.index(",")
                item = item_str[0: ind]
                imag = plt.imread(item)
                plt.imshow(imag)
                plt.show()
                item_str = item_str[ind+1:len(item_str)]
                print(item_str)
            to_save = input("Favorite item? y/n: ").upper()
            if to_save == "N":
                ret = db.collection_types["stack"].delete_one(r)
                print(ret)
            resp = input("\nGo ahead and wear outfit y/n?: ")
        else:
            get_outfit()
    if resp.upper() == "N":
        get_outfit()


# generate_all_from_scratch()
# update_accuracy()
# get_best(None, True)
# criteria = {"fashionable_likelihood": {"$gt": .70}, "distance": {"$gt": .2}}
# get_best(criteria, False, "stack")

# fit = db.collection_types["outfit"].find({"name": 193})
# for f in fit:
#     ret_code = db.collection_types["stack"].insert_one(f).inserted_id
#     print(ret_code)
get_outfit()

