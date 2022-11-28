import unittest

import pudb
import torch
from ridley.document_embeddings import (batch_riddle_candidates, embed,
                                        evaluate_riddle,
                                        mean_cosine_similarity,
                                        mean_euclidean_distance, score)
from transformers import RealmEmbedder, RealmScorer, RealmTokenizer


class TestDocumentEmbeddings(unittest.TestCase):
    def setUp(self):
        self.num_candidates = 2
        self.scorer = RealmScorer.from_pretrained(
            "google/realm-cc-news-pretrained-scorer", num_candidates=self.num_candidates
        )
        self.embedder = RealmEmbedder.from_pretrained(
            "google/realm-cc-news-pretrained-embedder"
        )
        self.tokenizer = RealmTokenizer.from_pretrained(
            "google/realm-cc-news-pretrained-embedder"
        )
        self.input_texts = ["How are you?", "What is the item in the picture?"]
        self.candidate_texts = [
            ["Hello world!", "Nice to meet you!"],
            ["A cute cat.", "An adorable dog"],
        ]

        self.input_file = "data/kaggle_riddles/riddles.csv"

        self.input_riddle = [
            (
                "Thirty white horses on a red hill, first they champ, then they "
                "stamp, then they stand still"
            )
        ]

    def test_tokenize(self):
        result = self.tokenizer(
            self.input_texts, return_tensors="pt", padding="longest"
        )
        self.assertIsNotNone(result)

    def test_embed(self):
        e = embed(self.embedder, self.tokenizer, self.input_texts)
        self.assertIsNotNone(e)

    def test_score(self):
        result = score(
            self.scorer, self.tokenizer, self.input_texts, self.candidate_texts
        )
        self.assertIsNotNone(result)

    def test_mean_cosine_similarity(self):
        result = mean_cosine_similarity(
            self.scorer, self.tokenizer, self.input_texts, self.candidate_texts
        )
        self.assertIsNotNone(result)
        # Test riddle
        batch = batch_riddle_candidates(self.input_file, self.num_candidates)
        batch = batch[:10]
        result = mean_cosine_similarity(
            self.scorer, self.tokenizer, self.input_riddle, batch
        )
        self.assertIsNotNone(result)

    def test_mean_euclidean_distance(self):
        result = mean_euclidean_distance(
            self.scorer, self.tokenizer, self.input_texts, self.candidate_texts
        )
        self.assertIsNotNone(result)

    def test_batch_riddle_candidates(self):
        # 2 candidates
        batch = batch_riddle_candidates(self.input_file, self.num_candidates)
        batch = batch[:10]
        self.assertIsInstance(batch, list)
        self.assertIsInstance(batch[0], list)
        self.assertIsInstance(batch[0][0], str)
        [self.assertEqual(len(b), self.num_candidates) for b in batch]

        # 8 candidates
        num_candidates = 8
        batch = batch_riddle_candidates(self.input_file, num_candidates)
        batch = batch[:10]
        self.assertIsInstance(batch, list)
        self.assertIsInstance(batch[0], list)
        self.assertIsInstance(batch[0][0], str)
        [self.assertEqual(len(b), num_candidates) for b in batch]

    def test_score_riddle(self):
        batched_riddles = batch_riddle_candidates(self.input_file, self.num_candidates)
        batched_riddles = batched_riddles[:10]
        result = score(self.scorer, self.tokenizer, self.input_riddle, batched_riddles)
        self.assertIsNotNone(result)

    def test_evaluate_riddle(self):
        result = evaluate_riddle(
            self.scorer,
            self.tokenizer,
            self.input_riddle,
            self.input_file,
            self.num_candidates,
        )
        self.assertIsNotNone(result)
        for key, value in result.items():
            self.assertIsInstance(value, float)


if __name__ == "__main__":
    unittest.main()
