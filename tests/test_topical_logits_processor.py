import unittest

from GPT2ForwardBackward.padded_encoder import Encoder
from ridley.logit_processors import TopicalLogitsProcessor
from ridley.riddle_generation import generate
from transformers import GenerationConfig


class TestTopicalLogitsProcessor(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Encoder()
        self.max_new_tokens = 10
        self.topics = ["Blood orange"]
        self.booster = 10
        self.topical_lp = TopicalLogitsProcessor(
            self.tokenizer, self.max_new_tokens, self.topics, self.booster
        )
        self.generation_config = GenerationConfig(
            max_new_tokens=20,
            eos_token_id=50256,
            bos_token_id=50256,
            do_sample=True,
        )

    def test_generate(self):
        expected_result = [
            " attribute attribute attribute attribute attribute attribute attribute attribute attribute attribute attribute attribute attribute attribute attribute attribute attribute attribute\n\n Hello world!"
        ]
        result = generate(
            "Hello world!",
            generation_config=self.generation_config,
            logits_processor=[self.topical_lp],
            backward=True,
            seed=100,
        )
        self.assertEqual(result, expected_result)

    def test_request_topics(self):
        expected_result = [
            "crimson",
            "blood orange",
            "red",
            "apple",
            "color",
            "blood",
            "stop",
            "apples",
            "wine",
            "colour",
            "fire",
            "attribute",
        ]

        input_word = self.topics[0]
        result = self.topical_lp.request_topics(input_word)
        self.assertEqual(result, expected_result)


if __name__ == "__main__":
    unittest.main()
