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


    def pull_squares_to_permut(self, new_item):
        1
        # TODO this is what you call to get all of the square of different type.



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

    db = database.collection_types["pair"]
    print(db.count_documents({}))
    tester = []
    a = {1, 2}

    b = {1, 2}

    tester.append(a)
    print(b in tester)



