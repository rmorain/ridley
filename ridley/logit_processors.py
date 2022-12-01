import requests
import torch
from transformers import LogitsProcessor


class RhymeLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        prior = self.rhyming_prior(input_ids, scores)
        scores = scores @ prior
        return scores

    def request_rhymes(self, word):
        response = requests.get(
            f"https://rhymebrain.com/talk?function=getRhymes&word={word}"
        )
        return response

    def rhyming_prior(self, input_ids, scores):
        # Decode input_ids
        text = self.tokenizer.decode(input_ids)
        # Get rhyming word
        word = text.split()[-1]
        response = self.request_rhymes(word)
        rhyming_words = [x["word"] for x in response.json()]
        rhyming_tokens = self.tokenizer(rhyming_words).input_ids
        flat_list = [item for sublist in rhyming_tokens for item in sublist]
        mask = torch.zeros(self.tokenizer.vocab_size)
        # Set rhyming tokens to 1
        mask[torch.tensor(flat_list)] = 1
        new_scores = scores * mask
        return new_scores
