import datetime
import cv2
import Directories


class DB:
    myclient = Directories.myclient
    databases = myclient.list_database_names()
    mydb = myclient["fashionphotos"]
    collection_types = Directories.collection_types

    # TODO this is what you call to get all of the square of different type.
    def pull_squares_to_permut(self, new_item):
        self

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
            "distance": distance,
            "date": datetime.datetime.utcnow(),

        }
        ret_code = collection.insert_one(image).inserted_id
        return ret_code

    # TODO add the params so the combo can be easily passed in here
    def favorite_a_combo(self, combo):
        self

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
        return ret_code

    def store_image(self, image, location, name, type_of, combination=None):
        full_path = location + name
        cv2.imwrite(full_path, image)
        return self.add_photo(item_type=type_of, location=location, name=name, combination=combination)
