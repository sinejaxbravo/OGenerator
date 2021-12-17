import pymongo


class DB:
    myclient = pymongo.MongoClient("mongodb://localhost:27017/")
    print(myclient.list_database_names())
    mydb = myclient["fashionphotos"]
    
    print(myclient.list_database_names())
