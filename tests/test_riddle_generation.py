import unittest

from ridley.riddle_generation import generate


class TestRiddleGeneration(unittest.TestCase):
    def setUp(self):
        self.prompt = "Here is a riddle that I really like: "
        self.num_results = 1
        self.seed = 42
        self.max_length = 30

    def test_gpt2(self):
        result = generate(
            prompt=self.prompt,
            num_return_sequences=self.num_results,
            seed=self.seed,
            max_length=self.max_length,
        )
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, str)


if __name__ == "__main__":
    unittest.main()
