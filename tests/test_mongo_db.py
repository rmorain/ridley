import unittest

from transformers import RealmEmbedder, RealmTokenizer

from ridley.ridley.MongoDBAccess import (get_database, get_collection, get_item, get_item_answer, add_item, drop_collection, hash_id)
from ridley.ridley.document_embeddings import embed

class TestMongoDB(unittest.TestCase):
    def setUp(self):
        self.uri = "mongodb+srv://Branking24:Password@cluster0.bwr9dko.mongodb.net/test"
        self.db = "test"
        self.collection = "testCol"
        self.test_riddle = ["Question", "Answer"]
        self.embedder = RealmEmbedder.from_pretrained(
            "google/realm-cc-news-pretrained-embedder"
        )
        self.tokenizer = RealmTokenizer.from_pretrained(
            "google/realm-cc-news-pretrained-embedder"
        )

    def test_get_database(self):
        db = get_database(self.db)
        self.assertIsNotNone(db)

    def test_get_collection(self):
        col = get_collection(self.db, self.collection)
        self.assertIsNotNone(col)

    def test_add_item(self):
        drop_collection(self.db, self.collection)
        e = embed(self.embedder, self.tokenizer, self.test_riddle)
        test_object = {"_id": hash_id(e), "test": "test", "answer": "answer"}
        add_item(self.db, self.collection, test_object)
        item = get_item(self.db, self.collection, e)
        self.assertIsNotNone(item)
        self.assertEqual(item, test_object)

    def test_embedding_hash(self):
        e = embed(self.embedder, self.tokenizer, self.test_riddle)
        h = hash_id(e)
        self.assertIsNotNone(h)

    def test_get_item_answer(self):
        drop_collection(self.db, self.collection)
        e = embed(self.embedder, self.tokenizer, self.test_riddle)
        test_object = {"_id": hash_id(e), "test": "test", "answer": "answer"}
        add_item(self.db, self.collection, test_object)
        item = get_item_answer(self.db, self.collection, e)
        self.assertIsNotNone(item)
        self.assertEqual(item, "answer")


if __name__ == "__main__":
    unittest.main()