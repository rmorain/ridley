import pickle
import unittest

from ridley.riddle_generation import (generate, generate_backward,
                                      generate_rhyming_lines)


class TestRiddleGeneration(unittest.TestCase):
    def setUp(self):
        self.prompt = "Here is a riddle that I really like: "
        self.num_results = 1
        self.seed = 42
        self.max_length = 30

    def test_generate(self):
        result = generate(
            prompt=self.prompt,
            num_return_sequences=self.num_results,
            seed=self.seed,
            max_length=self.max_length,
        )
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, str)

    def test_generate_rhyming_lines(self):
        result = generate_rhyming_lines("Have you heard about the man from Peru?")
        self.assertIsNotNone(result)

    def test_generate_rhyming_lines_backward(self):
        # result = generate_rhyming_lines_backward(
        # "Have you heard about the man from Peru"
        # )
        # self.assertIsNotNone(result)
        pass

    def test_generate_backward(self):
        with open("tests/data/generate_backward_result.pkl", "rb") as f:
            correct_output = pickle.load(f)
            result = generate_backward(self.prompt)
            self.assertEqual(result, correct_output)


if __name__ == "__main__":
    unittest.main()
