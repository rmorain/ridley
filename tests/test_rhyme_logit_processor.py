import pickle
import unittest

import requests
import torch
from GPT2ForwardBackward.padded_encoder import Encoder
from ridley.logit_processors import BackwardsRhymeLogitsProcessor
from ridley.riddle_generation import generate
from transformers import GenerationConfig


class TestRhymeLogitsProcessor(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Encoder()
        self.prompt = "I live in a town"
        self.num_results = 1
        self.seed = 42
        self.generation_config = GenerationConfig(
            max_new_tokens=20,
            eos_token_id=50256,
            bos_token_id=50256,
            do_sample=True,
        )

        self.rhyme_lp = BackwardsRhymeLogitsProcessor(
            self.tokenizer, self.generation_config.max_new_tokens
        )

    def test_generate(self):
        result = generate(
            self.prompt,
            generation_config=self.generation_config,
            seed=self.seed,
            logits_processor=[self.rhyme_lp],
            backward=True,
        )
        self.assertIsNotNone(result)

    def test_request_rhymes(self):
        result = self.rhyme_lp.request_rhymes("hello")
        self.assertIsInstance(result, requests.models.Response)

    def test_rhyming_prior(self):
        input_ids = self.tokenizer("Hello world", return_tensors="pt").input_ids
        scores = torch.rand(self.tokenizer.vocab_size)
        result, _ = self.rhyme_lp.rhyming_prior(input_ids, scores)
        self.assertIsNotNone(result)
        self.assertEqual(self.tokenizer.vocab_size, len(result))

    def test_select_rhyming_word(self):
        pass


if __name__ == "__main__":
    unittest.main()
