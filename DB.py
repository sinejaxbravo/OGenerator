import datetime

import cv2
import numpy as np
import pymongo
import scipy
from scipy import stats
from pymongo.collection import Collection
from pymongo.message import query
from pymongo import MongoClient

import Directories
import NeuralNet
import UnsupervisedClustering


class DB:
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    databases = myclient.list_database_names()
    print(myclient.list_database_names())
    mydb = myclient["fashionphotos"]

    collection_types = {"outfit": mydb.outfit, "shoe": mydb.shoes, "coat": mydb.coat, "shirt": mydb.shirt,
                        "pant": mydb.pant, "short": mydb.short, "pair": mydb.pair}

    print(mydb.list_collection_names())



    def pull_squares_to_permut(self, new_item):
        1
        # TODO this is what you call to get all of the square of different type.

    def add_outfits(self, name, items, temperature, percentage, pairs, coat, distance=None):
        collection = self.collection_types["outfit"]
        combination = ""
        for x in items:
            combination += str(x) + ", "

        image = {
            "name": name,
            "combination": combination,
            "temperature": temperature,
            "fashionable_likelihood": percentage,
            "pairs": pairs,
            "coat": coat,
            "distance":distance,
            "date": datetime.datetime.utcnow(),

        }
        ret_code = collection.insert_one(image).inserted_id
        # print(collection)
        # print(ret_code)
        return ret_code

    def favorite_a_combo(self, combo):
        1
        # TODO add the params so the combo can be easily passed in here

    def add_photo(self, item_type, location, name, combination=None, temperature=None):
        # print(self.mydb.list_collection_names())
        collection = self.collection_types[item_type]
        print(f"DB: LOCATION {location}, NAME {name}")

        image = {
            "name": name,
            "location": location,
            "combination": combination,
            "temperature": temperature,
            "date": datetime.datetime.utcnow(),

        }
        ret_code = collection.insert_one(image).inserted_id
        # print(collection)
        # print(ret_code)
        return ret_code

    def store_image(self, image, location, name, type_of, combination=None):
        full_path = location + name
        cv2.imwrite(full_path, image)
        return self.add_photo(item_type=type_of, location=location, name=name, combination=combination)


class test:
    l = [[9, 8], [9, 88], [0, 4]]
    l = np.array(l)
    l = scipy.stats.zscore(l)
    print(scipy.stats.norm.cdf(l))

    # res = NeuralNet.fashion_CNN()
    # stats = UnsupervisedClustering.model(Directories.dir_5, res, "x")
    # x_mean = stats[0]
    # y_mean = stats[1]
    # database = DB()
    # outfit = database.collection_types["outfit"]
    #
    # path = Directories.path_pair
    # outfits = outfit.find()
    # for fit in outfits:
    #     temp = fit["pairs"]
    #     square = []
    #     for t in temp:
    #         square.append(f"{path}{t}")
    #     print(square)
    #     val = UnsupervisedClustering.model(square, res)
    #
    #     print(val)
    #
    #     percentile = scipy.stats.norm.cdf(val)
    #     print(f"Percent chance: {percentile}")
    #     new_percent = []
    #     for p in percentile:
    #         new_percent.append([p[0], 1. - p[1]])
    #     print(new_percent)
    #     percentile = np.average(percentile)
    #     new_percent = np.average(new_percent)
    #     print(percentile)
    #     print(new_percent)






