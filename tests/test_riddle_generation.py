import unittest

from GPT2ForwardBackward.padded_encoder import Encoder
from ridley.logit_processors import BackwardsRhymeLogitsProcessor
from ridley.riddle_generation import generate, generate_lines
from transformers import GenerationConfig


class TestRiddleGeneration(unittest.TestCase):
    def setUp(self):
        self.prompt = "Here is a riddle that I really like: "
        self.num_results = 1
        self.seed = 42
        self.max_length = 30
        self.generation_config = GenerationConfig(
            max_new_tokens=20,
            eos_token_id=50256,
            bos_token_id=50256,
            do_sample=True,
        )
        self.rhyme_lp = BackwardsRhymeLogitsProcessor(
            Encoder(), self.generation_config.max_new_tokens, do_sample=True
        )

    def test_generate(self):
        # No arguments
        result = generate("Hello World")
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, str)
        # Generate forward with generation config
        result = generate(
            self.prompt,
            generation_config=self.generation_config,
            seed=self.seed,
        )
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, str)
        # Generate backward
        result = generate(
            self.prompt,
            generation_config=self.generation_config,
            seed=self.seed,
            backward=True,
        )
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, str)

    def test_generate_lines(self):
        result = generate_lines(
            "Have you heard about the man from Peru?",
            seed=self.seed,
            generation_config=self.generation_config,
            num_lines=3,
        )
        self.assertIsNotNone(result)

    def test_generate_rhyming_lines(self):
        result = generate_lines(
            "Have you heard about the man from Peru?",
            seed=self.seed,
            generation_config=self.generation_config,
            num_lines=3,
            logits_processor=[self.rhyme_lp],
            backward=True,
        )
        self.assertIsNotNone(result)


if __name__ == "__main__":
    unittest.main()
