import numpy as np
import pandas as pd
import torch
from transformers import RealmEmbedder, RealmScorer, RealmTokenizer


def score(model, tokenizer, input_texts, candidate_texts):
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True)
    candidate_inputs = tokenizer.batch_encode_candidates(
        candidate_texts,
        max_length=30,
        return_tensors="pt",
        truncation=True,
        padding=True,
    )
    outputs = model(
        **inputs,
        candidate_input_ids=candidate_inputs.input_ids,
        candidate_attention_mask=candidate_inputs.attention_mask,
        candidate_token_type_ids=candidate_inputs.token_type_ids,
    )
    return outputs


def embed(embedder, tokenizer, input_texts):
    inputs = tokenizer(input_texts, return_tensors="pt", padding=True)
    result = embedder(
        **inputs,
        output_hidden_states=True,
    )
    return result.projected_score


def mean_cosine_similarity(scorer, tokenizer, input_texts, candidate_texts):
    results = score(scorer, tokenizer, input_texts, candidate_texts)
    relevance_scores = results.relevance_score
    return 1 - relevance_scores.mean().item()  # 1 - to minimize


def mean_euclidean_distance(scorer, tokenizer, input_texts, candidate_texts):
    results = score(scorer, tokenizer, input_texts, candidate_texts)
    input_embeddings = results.query_score
    candidate_embeddings = results.candidate_score
    return torch.norm(input_embeddings - candidate_embeddings, dim=-1).mean().item()


def batch_riddle_candidates(input_file, num_candidates):
    riddles = pd.read_csv(input_file)[:300]  # Max 300 rows
    riddles_list = list(riddles.QUESTIONS + " Answer: " + riddles.ANSWERS)
    # Must all be same length
    while len(riddles_list) % num_candidates != 0:
        riddles_list = riddles_list[:-1]
    batched_riddles = np.array_split(riddles_list, len(riddles_list) // num_candidates)
    batched_riddles = [list(x) for x in batched_riddles]
    return batched_riddles


def evaluate_riddle(scorer, tokenizer, input_riddle, candidate_file, num_candidates):
    batched_riddles = batch_riddle_candidates(candidate_file, num_candidates)
    batched_riddles = batched_riddles[:10]

    cosine_similarity = mean_cosine_similarity(
        scorer, tokenizer, input_riddle, batched_riddles
    )

    euclidean_distance = mean_euclidean_distance(
        scorer, tokenizer, input_riddle, batched_riddles
    )
    return {
        "cosine similarity": cosine_similarity,
        "euclidean distance": euclidean_distance,
    }


def score_riddle(
    scorer,
    tokenizer,
    input_riddle,
    candidate_file1,
    candidate_file2,
    num_candidates,
    lamda,
    metric,
):
    typicality = evaluate_riddle(
        scorer, tokenizer, input_riddle, candidate_file1, num_candidates
    )
    novelty = evaluate_riddle(
        scorer, tokenizer, input_riddle, candidate_file2, num_candidates
    )
    return (-1 * lamda * typicality[metric]) + (lamda - 1) * novelty[metric]


if __name__ == "__main__":

    scorer = RealmScorer.from_pretrained(
        "google/realm-cc-news-pretrained-scorer", num_candidates=2
    )
    embedder = RealmEmbedder.from_pretrained("google/realm-cc-news-pretrained-embedder")
    tokenizer = RealmTokenizer.from_pretrained(
        "google/realm-cc-news-pretrained-embedder"
    )
    # input_texts = ["How are you?", "What is the item in the picture?"]
    input_texts = [
        """
        San Jose, officially San José, is a major city in the U.S. state of California
        that is the cultural, financial, and political center of Silicon Valley and
        largest city in Northern California by both population and area.
        """,
        """
        The theory of relativity usually encompasses two interrelated theories by
        Albert Einstein: special relativity and general relativity, proposed and
        published in 1905 and 1915, respectively.
        """,
    ]
    # input_texts = ["How are you?"]
    candidate_texts = [
        ["Hello world!", "Nice to meet you!"],
        ["A cute cat.", "An adorable dog"],
    ]

    # result = score(model, tokenizer, input_texts, candidate_texts)
    result = embed(embedder, tokenizer, input_texts)
    cos = torch.nn.CosineSimilarity(dim=0)
    E = result.hidden_states[-1]
    e1 = E[0, 0]
    e2 = E[1, 0]
    sim = cos(e1, e2)
    print(sim)

    riddles = pd.read_csv("data/kaggle_riddles/riddles.csv")
    input_riddle = """
        Thirty white horses on a red hill, first they champ, then they stamp, then they
        stand still
        """
    riddles_result = score(scorer, tokenizer, input_riddle, candidate_texts)
