import pickle
import unittest

from ridley.riddle_generation import (generate, generate_rhyming_lines,
                                      generate_rhyming_lines_backward)
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

    def test_generate_rhyming_lines(self):
        result = generate_rhyming_lines("Have you heard about the man from Peru?")
        self.assertIsNotNone(result)

    def test_generate_rhyming_lines_backward(self):
        with open("tests/data/generate_rhyming_lines_backward.pkl", "rb") as f:
            correct_output = pickle.load(f)
            result = generate_rhyming_lines_backward(
                "Have you heard about the man from Peru"
            )
            self.assertEqual(result, correct_output)


if __name__ == "__main__":
    unittest.main()
