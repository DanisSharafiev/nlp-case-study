import numpy as np
import pandas as pd

from src.case_study.ranking import rank_blocks, rank_docs


def reciprocal_rank(ranked: list, relevant: set) -> float:
    for index, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / index
    return 0.0


def hit_at_k(ranked: list, relevant: set, k: int) -> float:
    return float(bool(set(ranked[:k]) & relevant))


def evaluate_methods(
    method_scores: dict[str, np.ndarray],
    query_ids: list[int],
    row_by_query: dict[int, int],
    block_ids: list[str],
    block_doc_ids: list[int],
    relevant_blocks: dict[int, set[str]],
    relevant_docs: dict[int, set[int]],
    k: int = 10,
    method_rankings: dict[str, dict[str, list[list]]] | None = None,
) -> pd.DataFrame:
    rows = []

    for method, scores in method_scores.items():
        block_rr = []
        doc_rr = []
        doc_hit = []

        for qid in query_ids:
            row_index = row_by_query[qid]
            block_ranking = rank_blocks(scores[row_index], block_ids)
            doc_ranking = rank_docs(scores[row_index], block_doc_ids)

            block_rr.append(reciprocal_rank(block_ranking, relevant_blocks.get(qid, set())))
            doc_rr.append(reciprocal_rank(doc_ranking, relevant_docs.get(qid, set())))
            doc_hit.append(hit_at_k(doc_ranking, relevant_docs.get(qid, set()), k))

        rows.append(
            {
                "method": method,
                "block_mrr": float(np.mean(block_rr)),
                "doc_mrr": float(np.mean(doc_rr)),
                f"doc_hit@{k}": float(np.mean(doc_hit)),
            }
        )

    for method, rankings in (method_rankings or {}).items():
        block_rr = []
        doc_rr = []
        doc_hit = []

        for qid in query_ids:
            row_index = row_by_query[qid]
            block_ranking = rankings["block_rankings"][row_index]
            doc_ranking = rankings["doc_rankings"][row_index]

            block_rr.append(reciprocal_rank(block_ranking, relevant_blocks.get(qid, set())))
            doc_rr.append(reciprocal_rank(doc_ranking, relevant_docs.get(qid, set())))
            doc_hit.append(hit_at_k(doc_ranking, relevant_docs.get(qid, set()), k))

        rows.append(
            {
                "method": method,
                "block_mrr": float(np.mean(block_rr)),
                "doc_mrr": float(np.mean(doc_rr)),
                f"doc_hit@{k}": float(np.mean(doc_hit)),
            }
        )

    return pd.DataFrame(rows).sort_values("doc_mrr", ascending=False).reset_index(drop=True)


def evaluate_by_overlap_group(
    method_scores: dict[str, np.ndarray],
    pairs: pd.DataFrame,
    test_query_ids: list[int],
    row_by_query: dict[int, int],
    block_ids: list[str],
    block_doc_ids: list[int],
    method_rankings: dict[str, dict[str, list[list]]] | None = None,
) -> pd.DataFrame:
    rows = []
    test_pairs = pairs[pairs["query_id"].isin(test_query_ids)]

    for group in ["full", "partial", "none"]:
        group_pairs = test_pairs[test_pairs["overlap_group"] == group]
        if group_pairs.empty:
            continue
        group_doc_ids: dict[int, set[int]] = {}
        group_block_ids: dict[int, set[str]] = {}
        for _, pair in group_pairs.iterrows():
            qid = int(pair["query_id"])
            group_doc_ids.setdefault(qid, set()).add(int(pair["doc_id"]))
            group_block_ids.setdefault(qid, set()).add(str(pair["block_id"]))
        group_query_ids = sorted(group_doc_ids)

        for method, scores in method_scores.items():
            block_rr = []
            doc_rr = []
            doc_hit = []

            for qid in group_query_ids:
                row_index = row_by_query[qid]
                block_ranking = rank_blocks(scores[row_index], block_ids)
                doc_ranking = rank_docs(scores[row_index], block_doc_ids)

                block_rr.append(reciprocal_rank(block_ranking, group_block_ids[qid]))
                doc_rr.append(reciprocal_rank(doc_ranking, group_doc_ids[qid]))
                doc_hit.append(hit_at_k(doc_ranking, group_doc_ids[qid], 10))

            rows.append(
                {
                    "overlap_group": group,
                    "n_pairs": int(len(group_pairs)),
                    "n_queries": int(len(group_query_ids)),
                    "method": method,
                    "block_mrr": float(np.mean(block_rr)),
                    "doc_mrr": float(np.mean(doc_rr)),
                    "doc_hit@10": float(np.mean(doc_hit)),
                }
            )

        for method, rankings in (method_rankings or {}).items():
            block_rr = []
            doc_rr = []
            doc_hit = []

            for qid in group_query_ids:
                row_index = row_by_query[qid]
                block_ranking = rankings["block_rankings"][row_index]
                doc_ranking = rankings["doc_rankings"][row_index]

                block_rr.append(reciprocal_rank(block_ranking, group_block_ids[qid]))
                doc_rr.append(reciprocal_rank(doc_ranking, group_doc_ids[qid]))
                doc_hit.append(hit_at_k(doc_ranking, group_doc_ids[qid], 10))

            rows.append(
                {
                    "overlap_group": group,
                    "n_pairs": int(len(group_pairs)),
                    "n_queries": int(len(group_query_ids)),
                    "method": method,
                    "block_mrr": float(np.mean(block_rr)),
                    "doc_mrr": float(np.mean(doc_rr)),
                    "doc_hit@10": float(np.mean(doc_hit)),
                }
            )

    return pd.DataFrame(rows).sort_values(
        ["overlap_group", "doc_mrr"],
        ascending=[True, False],
    ).reset_index(drop=True)
