from transformers import LogitsProcessor


class RhymeLogitsProcessor(LogitsProcessor):
    def __init__(self):
        pass

    def __call__(self, input_ids, scores):
        return scores
