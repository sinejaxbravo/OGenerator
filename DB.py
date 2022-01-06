import datetime

import cv2
import pymongo
from pymongo.collection import Collection
from pymongo.message import query


class DB:
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    databases = myclient.list_database_names()
    print(myclient.list_database_names())
    mydb = myclient["fashionphotos"]

    collection_types = {"outfit": mydb.outfit, "shoe": mydb.shoes, "coat": mydb.coat, "shirt": mydb.shirt,
                        "pant": mydb.pant, "short": mydb.short, "pair": mydb.pair}

    print(mydb.list_collection_names())

    def add_photo(self, item_type, location, name, combination=None):
        # print(self.mydb.list_collection_names())
        collection = self.collection_types[item_type]

        image = {
            "name": name,
            "location": location,
            "date": datetime.datetime.utcnow(),
            "combination": combination
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

    db = database.collection_types["pair"]
    print(db.count_documents({}))

