import datetime

import cv2
import pymongo
from pymongo.collection import Collection
from pymongo.message import query
from pymongo import MongoClient

import Directories
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

    def add_outfits(self, name, items, temperature, percentage, pairs, coat):
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
    database = DB()
    db = database.collection_types["shirt"]
    print(db.collection_types)
    path = Directories.path_pair
    outfits = db.find({})
    print(outfits)
    for fit in outfits:
        print(fit)
        temp = fit["name"]
        db.update_one(fit, {"$set":{"name":"fuckyou"}})
    # db.inventory.find({fashionable_likelihood: {$gt: 80}})




