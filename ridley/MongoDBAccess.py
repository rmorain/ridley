import csv

import pymongo


def get_database(name):
    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    uri = "mongodb+srv://Branking24:Password@cluster0.bwr9dko.mongodb.net/test"
    client = pymongo.MongoClient(uri)
    db = client[name]

    # Create the database for our example
    # (we will use the same database throughout the tutorial
    return db


def get_collection(db, name):
    db = get_database(db)

    return db[name]


def drop_collection(db, name):
    db = get_database(db)
    collection = db[name]
    collection.drop()


def add_item(db, name, item):
    collection = get_collection(db, name)
    collection.insert_one(item)
    return


def add_many_items(db, name, items):
    collection = get_collection(db, name)
    collection.insert_many(items)
    return


def read_csv(pathname):
    with open(pathname) as csvfile:
        csvReader = csv.reader(csvfile, delimiter=",")
        objects = []
        object = {"_id": "", "question": "", "answer": ""}
        i = 0
        for row in csvReader:
            if row[0] == "QUESTIONS":
                continue
            object = {"_id": hash_id(row[0]), "question": row[0], "answer": row[1]}
            i = i + 1
            objects.append(object)
        return objects


def hash_id(j):
    return hash(j)


def get_all_items(db, coll):
    collection = get_collection(db, coll)
    items = []
    for i in collection.find():
        items.append(i)
    return items


def get_item(db, coll, id):
    collection = get_collection(db, coll)
    all = collection.find({"_id": hash_id(id)})
    items = []
    for i in all:
        items.append(i)
    if len(items) > 1:
        return {"error": "multiple items with same id"}
    elif len(items) < 1:
        return {"error": "no items with that id"}
    else:
        return items[0]


def get_item_answer(db, coll, id):
    collection = get_collection(db, coll)
    all = collection.find({"_id": hash_id(id)})
    items = []
    for i in all:
        items.append(i)
    if len(items) > 1:
        return {"error": "multiple items with same id"}
    elif len(items) < 1:
        return {"error": "no items with that id"}
    else:
        return items[0]["answer"]


# drop_collection("riddles", "kriddles")

# add_many_items("riddles", "kriddles", read_csv("riddles.csv"))
