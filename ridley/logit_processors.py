import random

import requests
import torch
from transformers import LogitsProcessor


class RhymeLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rhyming_word_tokens = None
        self.rhyming_word_index = -1
        self.rhyming_scores = None

    def __call__(self, input_ids, scores):
        scores.squeeze_()
        if not self.rhyming_word_tokens or self.rhyming_word_index > (
            len(self.rhyming_word_tokens) - 1
        ):
            prior, rhyming_tokens = self.rhyming_prior(input_ids, scores)
            self.rhyming_scores = scores * prior
            self.rhyming_word_tokens = random.choice(rhyming_tokens)
            self.rhyming_word_index = 0
        new_scores = torch.full_like(scores, fill_value=float("-inf"))
        new_scores[self.rhyming_word_tokens[self.rhyming_word_index]] = 1
        self.rhyming_word_index += 1
        return new_scores.unsqueeze_(0)

    def request_rhymes(self, word):
        response = requests.get(
            f"https://rhymebrain.com/talk?function=getRhymes&word={word}"
        )
        return response

    def rhyming_prior(self, input_ids, scores):
        # Decode input_ids
        text = self.tokenizer.decode(input_ids[0])
        word = text.split()[-1]
        # Get rhyming word
        response = self.request_rhymes(word)
        # Add space before word to correctly tokenize.
        rhyming_words = [" " + x["word"] for x in response.json()]
        if len(rhyming_words) > 0:
            rhyming_tokens = self.tokenizer(rhyming_words).input_ids
            flat_list = [item for sublist in rhyming_tokens for item in sublist]
            mask = torch.zeros(self.tokenizer.vocab_size)
            # Set rhyming tokens to 1
            mask[torch.tensor(flat_list)] = 1
        # If nothing rhymes
        else:
            mask = torch.ones(self.tokenizer.vocab_size)
        return mask, rhyming_tokens
