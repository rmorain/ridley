import numpy as np
import requests
import torch
from transformers import LogitsProcessor


class RhymeLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rhyming_word_tokens = None
        self.rhyming_word_index = -1
        self.rhyming_scores = None
        self.sequence_length = -1
        self.prompt_len = None
        self.call_counter = 0

    def __call__(self, input_ids, scores):
        scores.squeeze_()
        if not self.rhyming_word_tokens or self.rhyming_word_index > (
            len(self.rhyming_word_tokens) - 1
        ):
            self.prompt_len = len(input_ids[0])
            prior, rhyming_tokens = self.rhyming_prior(input_ids, scores)
            self.rhyming_scores = scores * prior
            self.rhyming_word_tokens = self.select_rhyming_word(rhyming_tokens, scores)
            self.rhyming_word_index = 0
        if len(input_ids[0]) - self.prompt_len >= self.max_length - len(
            self.rhyming_word_tokens
        ):
            new_scores = torch.full_like(scores, fill_value=float("-inf"))
            new_scores[self.rhyming_word_tokens[self.rhyming_word_index]] = 1
            self.rhyming_word_index += 1
        else:
            return scores.unsqueeze_(0)
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

    def select_rhyming_word(self, rhyming_tokens, scores):
        word_scores = []
        for word in rhyming_tokens:
            avg_word_score = scores[word].mean().item()
            word_scores.append(avg_word_score)
        best_rhyming_word = rhyming_tokens[np.argmax(word_scores)]
        return best_rhyming_word


class BackwardsRhymeLogitsProcessor(RhymeLogitsProcessor):
    def __init__(self, tokenizer, max_new_tokens):
        super().__init__(tokenizer, max_new_tokens)

    def __call__(self, input_ids, scores):
        self.call_counter += 1
        scores.squeeze_()
        if not self.rhyming_word_tokens:
            forward_ids = [input_ids[0].tolist()[::-1]]
            prior, rhyming_tokens = self.rhyming_prior(forward_ids, scores)
            self.rhyming_word_tokens = self.select_rhyming_word(rhyming_tokens, scores)
            self.rhyming_word_tokens = self.rhyming_word_tokens[::-1]
            self.rhyming_word_index = 0
        if self.rhyming_word_index < len(self.rhyming_word_tokens):
            new_scores = torch.full_like(scores, fill_value=float("-inf"))
            new_scores[self.rhyming_word_tokens[self.rhyming_word_index]] = 1
            self.rhyming_word_index += 1
            return new_scores.unsqueeze_(0)
        if self.call_counter >= self.max_length:
            # Reinitialize for next line
            super().__init__(self.tokenizer, self.max_length)

        return scores.unsqueeze_(0)

    def rhyming_prior(self, input_ids, scores):
        # Decode input_ids
        text = self.tokenizer.decode(input_ids[0])
        word = text.split()[-1]
        # Get rhyming word
        response = self.request_rhymes(word)
        # Add space before word to correctly tokenize.
        rhyming_words = [" " + x["word"] for x in response.json()]
        if len(rhyming_words) > 0:
            rhyming_tokens = [self.tokenizer.encode(word) for word in rhyming_words]
            flat_list = [item for sublist in rhyming_tokens for item in sublist]
            mask = torch.zeros(self.tokenizer.encoder.vocab_size)
            # Set rhyming tokens to 1
            mask[torch.tensor(flat_list)] = 1
        # If nothing rhymes
        else:
            mask = torch.ones(self.tokenizer.encoder.vocab_size)
        return mask, rhyming_tokens
