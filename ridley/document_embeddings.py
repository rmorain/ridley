import torch
from transformers import RealmScorer, RealmTokenizer


def score():
    tokenizer = RealmTokenizer.from_pretrained("google/realm-cc-news-pretrained-scorer")
    model = RealmScorer.from_pretrained(
        "google/realm-cc-news-pretrained-scorer", num_candidates=2
    )

    # batch_size = 2, num_candidates = 2
    input_texts = ["How are you?", "What is the item in the picture?"]
    candidate_texts = [
        ["Hello world!", "Nice to meet you!"],
        ["A cute cat.", "An adorable dog"],
    ]

    inputs = tokenizer(input_texts, return_tensors="pt")
    print(inputs)
