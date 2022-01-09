import asyncio
import math
import os
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
from main import make_pairs, make_squares

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

    # shoes = make_squares(paths["shoe"], "shoe")

    shirt = make_squares(paths["shirt"], "shirt")

    coat = make_squares(paths["coat"], "coat")

    with_c = []
    without_c = []
    c_num = 1
    wc = []
    woc = []
    outfit_num = 0
    for p in pants:
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


# def get_stats():
#     stats = UnsupervisedClustering.model(Directories.dir_zip, res, "x")
#     # stats = UnsupervisedClustering.model(Directories.dir_zip, res, "x")
#     print(stats)
#     return stats


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


def Affinity(data, labels):
    X = StandardScaler().fit_transform(data)
    model = AffinityPropagation(damping=0.9, random_state=None)
    # fit the model
    model.fit(X)
    # assign a cluster to each example
    yhat = model.predict(X)
    # retrieve unique clusters
    clusters = np.unique(yhat)
    # print(clusters)

    return X, labels
    # return np.average(X[:, 0]), np.average(X[:, 1])


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
    l = [[9, 8], [-9, -88], [0, 4]]
    l = np.array(l)
    print(scipy.stats.zscore(l[:, 1]))
    res = UnsupervisedClustering.train()
    database = DB()
    outfit = database.collection_types["outfit"]

    pairs, pair_feats = features_lists(res)

    # X, labels = Affinity(pair_feats, pairs)
    X = StandardScaler().fit_transform(pair_feats)
    print(db.collection_types)
    path = Directories.path_pair
    outfits = outfit.find()
    l_label = pairs.tolist()
    print(pairs)
    print(len(pairs))
    print(len(X))
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


def get_best():
    database = DB()
    outfit = database.collection_types["outfit"]
    outfits = outfit.find().sort({"distance": 1})
    for fit in outfits:
        print(fit["name"], fit["distance"])
        time.sleep(4)

# generate_all_from_scratch()
update_accuracy()
