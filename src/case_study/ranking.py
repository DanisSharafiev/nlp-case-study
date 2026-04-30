import numpy as np


def rank_blocks(scores: np.ndarray, block_ids: list[str]) -> list[str]:
    order = np.argsort(-scores)
    return [block_ids[i] for i in order]


def rank_docs(scores: np.ndarray, block_doc_ids: list[int]) -> list[int]:
    best_scores: dict[int, float] = {}
    for score, doc_id in zip(scores, block_doc_ids):
        if doc_id not in best_scores or score > best_scores[doc_id]:
            best_scores[doc_id] = float(score)
    return [
        doc_id
        for doc_id, _ in sorted(best_scores.items(), key=lambda item: item[1], reverse=True)
    ]


def interleave_unique(first: list, second: list) -> list:
    rankings = [first, second]
    positions = [0, 0]
    seen = set()
    merged = []
    turn = 0
    target_len = len(set(first) | set(second))

    while len(merged) < target_len:
        ranking = rankings[turn]
        while positions[turn] < len(ranking) and ranking[positions[turn]] in seen:
            positions[turn] += 1

        if positions[turn] < len(ranking):
            item = ranking[positions[turn]]
            merged.append(item)
            seen.add(item)
            positions[turn] += 1
        elif positions[0] >= len(first) and positions[1] >= len(second):
            break

        turn = 1 - turn

    return merged


def build_interleave_rankings(
    tfidf_scores: np.ndarray,
    bert_scores: np.ndarray,
    block_ids: list[str],
    block_doc_ids: list[int],
) -> dict[str, dict[str, list[list]]]:
    block_rankings = []
    doc_rankings = []

    for row_index in range(tfidf_scores.shape[0]):
        tfidf_block_ranking = rank_blocks(tfidf_scores[row_index], block_ids)
        bert_block_ranking = rank_blocks(bert_scores[row_index], block_ids)
        tfidf_doc_ranking = rank_docs(tfidf_scores[row_index], block_doc_ids)
        bert_doc_ranking = rank_docs(bert_scores[row_index], block_doc_ids)

        block_rankings.append(interleave_unique(tfidf_block_ranking, bert_block_ranking))
        doc_rankings.append(interleave_unique(tfidf_doc_ranking, bert_doc_ranking))

    return {
        "hybrid_interleave": {
            "block_rankings": block_rankings,
            "doc_rankings": doc_rankings,
        }
    }
