import numpy as np
import pandas as pd

from src.case_study.config import BLOCK_SIZE, STRIDE, SYNTHETIC_FULL_QUERIES
from src.case_study.text import split_sentences, synthetic_query_from_block, token_overlap


def make_blocks(
    documents: pd.DataFrame,
    block_size: int = BLOCK_SIZE,
    stride: int = STRIDE,
) -> pd.DataFrame:
    rows = []

    for _, row in documents.iterrows():
        doc_id = int(row["doc_id"])
        sentences = split_sentences(row["text"])
        if not sentences:
            continue

        if len(sentences) <= block_size:
            starts = [0]
        else:
            starts = list(range(0, len(sentences) - block_size + 1, stride))

        for block_no, start in enumerate(starts):
            chunk = sentences[start : start + block_size]
            rows.append(
                {
                    "block_id": f"{doc_id}_b{block_no:03d}",
                    "doc_id": doc_id,
                    "text": " ".join(chunk),
                    "start_sentence": start,
                    "end_sentence": start + len(chunk),
                }
            )

    return pd.DataFrame(rows)


def qrels_by_query(qrels: pd.DataFrame) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for _, row in qrels.iterrows():
        result.setdefault(int(row["query_id"]), set()).add(int(row["doc_id"]))
    return result


def add_synthetic_full_queries(
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    blocks: pd.DataFrame,
    n_queries: int = SYNTHETIC_FULL_QUERIES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    queries = queries.copy()
    qrels = qrels.copy()
    queries["source"] = queries.get("source", "cisi")

    candidates = blocks.copy()
    candidates["synthetic_query"] = candidates["text"].map(synthetic_query_from_block)
    candidates = candidates[candidates["synthetic_query"].map(lambda text: len(text.split()) >= 6)]
    candidates = candidates.drop_duplicates("doc_id").reset_index(drop=True)

    n_queries = min(n_queries, len(candidates))
    selected = candidates.iloc[np.linspace(0, len(candidates) - 1, n_queries, dtype=int)]
    start_query_id = int(queries["query_id"].max()) + 1

    synthetic_queries = pd.DataFrame(
        {
            "query_id": range(start_query_id, start_query_id + n_queries),
            "text": selected["synthetic_query"].tolist(),
            "source": "synthetic_full",
        }
    )
    synthetic_qrels = pd.DataFrame(
        {
            "query_id": synthetic_queries["query_id"].tolist(),
            "doc_id": selected["doc_id"].astype(int).tolist(),
            "relevance": 1,
        }
    )

    return (
        pd.concat([queries, synthetic_queries], ignore_index=True),
        pd.concat([qrels, synthetic_qrels], ignore_index=True),
    )


def relevant_pair_table(
    queries: pd.DataFrame,
    qrels: pd.DataFrame,
    blocks: pd.DataFrame,
) -> pd.DataFrame:
    query_text = dict(zip(queries["query_id"].astype(int), queries["text"]))
    query_source = dict(zip(queries["query_id"].astype(int), queries["source"]))
    blocks_by_doc = {doc_id: group for doc_id, group in blocks.groupby("doc_id")}
    rows = []

    for qid, doc_ids in qrels_by_query(qrels).items():
        query = query_text[qid]
        for doc_id in doc_ids:
            doc_blocks = blocks_by_doc.get(doc_id)
            if doc_blocks is None or doc_blocks.empty:
                continue
            scores = doc_blocks["text"].map(lambda text: token_overlap(query, text))
            best_row = doc_blocks.iloc[int(scores.to_numpy().argmax())]
            overlap = float(scores.max())
            if overlap == 0.0:
                group = "none"
            elif overlap >= 1.0:
                group = "full"
            else:
                group = "partial"
            rows.append(
                {
                    "query_id": qid,
                    "doc_id": doc_id,
                    "block_id": str(best_row["block_id"]),
                    "query_source": query_source[qid],
                    "overlap": overlap,
                    "overlap_group": group,
                }
            )

    return pd.DataFrame(rows)
