import numpy as np


def normalize_score_rows(scores: np.ndarray) -> np.ndarray:
    minimum = scores.min(axis=1, keepdims=True)
    maximum = scores.max(axis=1, keepdims=True)
    span = maximum - minimum
    return np.divide(scores - minimum, span, out=np.zeros_like(scores), where=span > 0)


def build_method_scores(
    tfidf_scores: np.ndarray,
    bert_scores: np.ndarray,
    coverages: np.ndarray,
    alpha: float,
    fallback_threshold: float,
) -> dict[str, np.ndarray]:
    tfidf_norm = normalize_score_rows(tfidf_scores)
    bert_norm = normalize_score_rows(bert_scores)
    return {
        "tfidf": tfidf_scores,
        "bert": bert_scores,
        "hybrid_fallback": np.where(
            coverages[:, None] < fallback_threshold,
            bert_scores,
            tfidf_scores,
        ),
        "hybrid_weighted": alpha * tfidf_norm + (1.0 - alpha) * bert_norm,
    }
