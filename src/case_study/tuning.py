from typing import Iterable

import numpy as np

from src.case_study.hybrids import build_method_scores
from src.case_study.metrics import evaluate_methods


def tune_alpha(
    tfidf_scores: np.ndarray,
    bert_scores: np.ndarray,
    val_query_ids: list[int],
    row_by_query: dict[int, int],
    block_ids: list[str],
    block_doc_ids: list[int],
    relevant_blocks: dict[int, set[str]],
    relevant_docs: dict[int, set[int]],
    candidates: Iterable[float] = np.linspace(0, 1, 11),
) -> float:
    best_alpha = 0.0
    best_mrr = -1.0
    coverages = np.ones(tfidf_scores.shape[0])

    for alpha in candidates:
        scores = build_method_scores(tfidf_scores, bert_scores, coverages, float(alpha), 0.0)
        metrics = evaluate_methods(
            {"hybrid_weighted": scores["hybrid_weighted"]},
            val_query_ids,
            row_by_query,
            block_ids,
            block_doc_ids,
            relevant_blocks,
            relevant_docs,
        )
        mrr = float(metrics.loc[0, "doc_mrr"])
        if mrr > best_mrr:
            best_alpha = float(alpha)
            best_mrr = mrr

    return best_alpha


def tune_fallback_threshold(
    tfidf_scores: np.ndarray,
    bert_scores: np.ndarray,
    coverages: np.ndarray,
    val_query_ids: list[int],
    row_by_query: dict[int, int],
    block_ids: list[str],
    block_doc_ids: list[int],
    relevant_blocks: dict[int, set[str]],
    relevant_docs: dict[int, set[int]],
    candidates: Iterable[float] = (0.25, 0.5, 0.75, 1.01),
) -> float:
    best_threshold = 0.5
    best_mrr = -1.0

    for threshold in candidates:
        scores = build_method_scores(tfidf_scores, bert_scores, coverages, 0.5, float(threshold))
        metrics = evaluate_methods(
            {"hybrid_fallback": scores["hybrid_fallback"]},
            val_query_ids,
            row_by_query,
            block_ids,
            block_doc_ids,
            relevant_blocks,
            relevant_docs,
        )
        mrr = float(metrics.loc[0, "doc_mrr"])
        if mrr > best_mrr:
            best_threshold = float(threshold)
            best_mrr = mrr

    return best_threshold
