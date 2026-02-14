import re
import string
from collections import Counter
from typing import Union

def normalize_answer(s: Union[str]):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))

def f1_score(prediction, ground_truth, normalize=True):
    if normalize:
        prediction_tokens = normalize_answer(prediction).split()
        ground_truth_tokens = normalize_answer(ground_truth).split()
    else:
        prediction_tokens = prediction.split()
        ground_truth_tokens = ground_truth.split()

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)

def exact_match(prediction, ground_truth, normalize=True):
    if normalize:
        return normalize_answer(prediction) == normalize_answer(ground_truth)
    return prediction == ground_truth
