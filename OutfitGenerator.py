import asyncio

import cv2
import numpy as np
import scipy.stats
from matplotlib import pyplot as plt

import Directories
import UnsupervisedClustering
from Weather import extract_weather
import Directories as dir
from DB import DB
from main import make_pairs, make_squares
res = UnsupervisedClustering.train()
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

    shoes = make_squares(paths["shoe"], "shoe")

    shirt = make_squares(paths["shirt"], "shirt")

    coat = make_squares(paths["coat"], "coat")



    with_c = []
    without_c = []
    c_num = 1
    wc = []
    woc = []
    outfit_num = 0
    for p in pants:
        for s in shoes:
            retps, ps = stitch(p, s, c_num, output_path_to_use)
            for sh in shirt:
                retpsh, psh = stitch(p, sh, c_num + 1, output_path_to_use)
                retpssh, ssh = stitch(s, sh, c_num + 3, output_path_to_use)
                for c in coat:
                    wc = [retps, retpsh, retpssh]
                    woc = [retps, retpsh, retpssh]
                    ret, pc = stitch(p, c, c_num + 2, output_path_to_use)
                    wc.append(ret)
                    ret, sc = stitch(s, c, c_num + 4, output_path_to_use)
                    wc.append(ret)
                    ret, shc = stitch(sh, c, c_num + 5, output_path_to_use)
                    wc.append(ret)
                    c_num += 6
                    db.add_outfits(outfit_num, [p[0], s[0], sh[0]], "0", 0, woc, "no")
                    db.add_outfits(outfit_num+1, [p[0], s[0], sh[0], c[0]], "0", 0, wc, "yes")
                    outfit_num += 2



def get_stats():
    print("Entered")
    stats = UnsupervisedClustering.model(Directories.dir_pred, res, "x")
    print(stats)
    return stats




def update_accuracy():
    stats = get_stats()
    x_mean = stats[0]
    y_mean = stats[1]
    database = DB()
    outfit = database.collection_types["outfit"]
    print(db.collection_types)
    path = Directories.path_pair
    outfits = outfit.find()
    for fit in outfits:
        temp = fit["pairs"]
        square = []
        for t in temp:
            square.append(f"{path}{t}")
        print(square)
        val = UnsupervisedClustering.model(square, res)
        percentile = scipy.stats.norm.cdf(val)
        print(percentile, "\n")
        percentile = np.average(percentile)
        outfit.update_one(fit, {"$set": {"fashionable_likelihood": percentile}})
        print(percentile, " saved!", "\n\n\n")


# get_stats()
update_accuracy()
