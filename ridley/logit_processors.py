import string

import numpy as np
import requests
import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

from ridley.ConceptNetAPiAccess import GetAllCommonNeighbors, GetSecondDegreeNeighborsWithPath



class RhymeLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, max_length, do_sample=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.rhyming_word_tokens = None
        self.rhyming_word_index = -1
        self.rhyming_scores = None
        self.sequence_length = -1
        self.prompt_len = None
        self.call_counter = 0
        self.do_sample = do_sample

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
        rhyming_words = []
        for x in response.json():
            # Require perfect rhymes
            if x["score"] >= 300:
                rhyming_words.append(x["word"])

        if len(rhyming_words) > 0:
            rhyming_tokens = self.tokenizer(
                rhyming_words, add_prefix_space=True
            ).input_ids
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
        if self.do_sample:
            word_scores = torch.Tensor(word_scores)
            probabilities = F.softmax(word_scores, dim=0)
            best_rhyming_word = rhyming_tokens[
                torch.multinomial(probabilities, num_samples=1)
            ]
        else:
            best_rhyming_word = rhyming_tokens[np.argmax(word_scores)]
        return best_rhyming_word


class BackwardsRhymeLogitsProcessor(RhymeLogitsProcessor):
    def __init__(self, tokenizer, max_new_tokens, do_sample=False):
        super().__init__(tokenizer, max_new_tokens, do_sample=do_sample)

    def __call__(self, input_ids, scores):
        self.call_counter += 1
        scores.squeeze_()
        if not self.rhyming_word_tokens:
            prior, rhyming_tokens = self.rhyming_prior(input_ids, scores)
            self.rhyming_word_tokens = self.select_rhyming_word(rhyming_tokens, scores)
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
        # Remove punctuation
        word = word.translate(str.maketrans("", "", string.punctuation))
        # Get rhyming word
        response = self.request_rhymes(word)
        # Add space before word to correctly tokenize.
        rhyming_words = []
        for x in response.json():
            # Require perfect rhymes
            if x["score"] >= 200:
                rhyming_words.append(x["word"])
        if len(rhyming_words) > 0:
            rhyming_tokens = []
            flat_list = []
            for word in rhyming_words:
                tokens = (
                    self.tokenizer(word, add_prefix_space=True, return_tensors="pt")
                    .input_ids.squeeze()
                    .tolist()
                )
                if type(tokens) == int:
                    rhyming_tokens.append([tokens])
                    flat_list.append(tokens)
                    continue
                rhyming_tokens.append(tokens)
                for item in tokens:
                    flat_list.append(item)

            mask = torch.zeros(self.tokenizer.vocab_size)
            # Set rhyming tokens to 1
            mask[torch.tensor(flat_list)] = 1
        # If nothing rhymes
        else:
            mask = torch.ones(self.tokenizer.vocab_size)
        return mask, rhyming_tokens


class TopicalPriorLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, max_length, topics, booster):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.booster = booster
        self.pre_retrieved = {}
        self.topics = topics

    def __call__(self, input_ids, scores):
        scores.squeeze_()
        max_score = max(scores)
        for w in self.topics:
            t = self.request_topics(w)
            for ind in t:
                t_tokens = self.tokenizer(ind, return_tensors="pt").input_ids
                for token in t_tokens:
                    scores[token] += ((max_score - scores[token]) * 0.75)

        return scores.unsqueeze_(0)

    def request_topics(self, input_word):
        if input_word not in self.pre_retrieved:
            response = GetAllCommonNeighbors(input_word)
            response = [" " + word for word in response]

            self.pre_retrieved[input_word] = response
        else:
            response = self.pre_retrieved[input_word]

        return response

    def request_topic(self, input_word):
        if input_word not in self.pre_retrieved:
            response = GetSecondDegreeNeighborsWithPath(input_word)[0]
            all = [response[0]]
            self.pre_retrieved[input_word] = all
        else:
            all = self.pre_retrieved[input_word]
        return all

    def get_topic_context_scores(self, topics):
        for topic in topics:
            return topic