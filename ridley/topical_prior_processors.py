import time

import numpy as np
import requests
import torch
from ridley.ridley.ConceptNetAPiAccess import GetAllCommonNeighbors, GetCommonNeighbor
from transformers import LogitsProcessor


class TopicalPriorLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, max_length, topics, booster):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.booster = booster
        self.pre_retrieved = {}
        self.topics = topics

    def __call__(self, input_ids, scores):
        scores.squeeze_()
        for w in self.topics:
            t = self.request_topics(w)
            for ind in t:
                t_tokens = self.tokenizer(ind).input_ids
                for token in t_tokens:
                    scores[token] += self.booster
        '''sentence = self.tokenizer.decode(input_ids[0])
        words = sentence.strip().split(" ")
        for w in words:
            #print(w)
            t = self.request_topics(w)
            for ind in t:
                t_tokens = self.tokenizer(ind).input_ids
                for token in t_tokens:
                    scores[token] += self.booster'''

        return scores.unsqueeze_(0)

    def request_topics(self, input_word):
        if input_word not in self.pre_retrieved:
            response = GetAllCommonNeighbors(input_word)

            self.pre_retrieved[input_word] = response
        else:
            response = self.pre_retrieved[input_word]

        return response

    def request_topic(self, input_word):
        if input_word not in self.pre_retrieved:
            response = GetCommonNeighbor(input_word)
            all = [response.split('/')[-1]]
            self.pre_retrieved[input_word] = all
        else:
            all = self.pre_retrieved[input_word]
        return all

