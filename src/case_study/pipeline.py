import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.case_study.config import SYNTHETIC_FULL_QUERIES
from src.case_study.data import (
    add_synthetic_full_queries,
    make_blocks,
    qrels_by_query,
    relevant_pair_table,
)
from src.case_study.embeddings import load_embeddings
from src.case_study.hybrids import build_method_scores
from src.case_study.metrics import evaluate_by_overlap_group, evaluate_methods
from src.case_study.ranking import build_interleave_rankings
from src.case_study.text import clean_text
from src.case_study.tfidf import fit_tfidf, query_vocab_coverage, tfidf_query_scores
from src.case_study.tuning import tune_alpha, tune_fallback_threshold


def run_pipeline(
    data_dir: Path | str = "data/processed",
    artifacts_dir: Path | str = "artifacts",
    synthetic_full_queries: int = SYNTHETIC_FULL_QUERIES,
) -> dict:
    data_dir = Path(data_dir)
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    documents = pd.read_csv(data_dir / "documents.csv")
    queries = pd.read_csv(data_dir / "queries.csv")
    qrels = pd.read_csv(data_dir / "qrels.csv")

    documents["text"] = documents["text"].map(clean_text)
    queries["text"] = queries["text"].map(clean_text)

    blocks = make_blocks(documents)
    blocks.to_csv(artifacts_dir / "blocks.csv", index=False)
    queries, qrels = add_synthetic_full_queries(
        queries,
        qrels,
        blocks,
        n_queries=synthetic_full_queries,
    )
    queries.to_csv(artifacts_dir / "queries_augmented.csv", index=False)
    qrels.to_csv(artifacts_dir / "qrels_augmented.csv", index=False)

    vectorizer, block_tfidf = fit_tfidf(blocks)
    relevant_docs = qrels_by_query(qrels)
    pairs = relevant_pair_table(queries, qrels, blocks)
    pairs.to_csv(artifacts_dir / "relevant_pairs.csv", index=False)
    relevant_blocks: dict[int, set[str]] = {}
    for _, row in pairs.iterrows():
        relevant_blocks.setdefault(int(row["query_id"]), set()).add(str(row["block_id"]))

    query_text_by_id = dict(zip(queries["query_id"].astype(int), queries["text"]))
    query_source_by_id = dict(zip(queries["query_id"].astype(int), queries["source"]))
    query_ids = sorted(set(relevant_docs) & set(relevant_blocks) & set(query_text_by_id))
    query_texts = [query_text_by_id[qid] for qid in query_ids]
    row_by_query = {qid: index for index, qid in enumerate(query_ids)}

    block_ids = blocks["block_id"].astype(str).tolist()
    block_doc_ids = blocks["doc_id"].astype(int).tolist()

    tfidf_scores = tfidf_query_scores(vectorizer, block_tfidf, query_texts)
    block_embeddings = load_embeddings(artifacts_dir / "block_embeddings.npy")
    query_embeddings = load_embeddings(artifacts_dir / "query_embeddings.npy")
    bert_scores = query_embeddings @ block_embeddings.T
    coverages = np.array([query_vocab_coverage(text, vectorizer) for text in query_texts])

    val_query_ids = query_ids[::4]
    test_query_ids = [qid for qid in query_ids if qid not in set(val_query_ids)]

    alpha = tune_alpha(
        tfidf_scores,
        bert_scores,
        val_query_ids,
        row_by_query,
        block_ids,
        block_doc_ids,
        relevant_blocks,
        relevant_docs,
    )
    fallback_threshold = tune_fallback_threshold(
        tfidf_scores,
        bert_scores,
        coverages,
        val_query_ids,
        row_by_query,
        block_ids,
        block_doc_ids,
        relevant_blocks,
        relevant_docs,
    )

    method_scores = build_method_scores(
        tfidf_scores,
        bert_scores,
        coverages,
        alpha=alpha,
        fallback_threshold=fallback_threshold,
    )
    method_rankings = build_interleave_rankings(
        tfidf_scores,
        bert_scores,
        block_ids,
        block_doc_ids,
    )
    overall = evaluate_methods(
        method_scores,
        test_query_ids,
        row_by_query,
        block_ids,
        block_doc_ids,
        relevant_blocks,
        relevant_docs,
        method_rankings=method_rankings,
    )
    by_group = evaluate_by_overlap_group(
        method_scores,
        pairs,
        test_query_ids,
        row_by_query,
        block_ids,
        block_doc_ids,
        method_rankings=method_rankings,
    )

    overall.to_csv(artifacts_dir / "overall_metrics.csv", index=False)
    by_group.to_csv(artifacts_dir / "overlap_metrics.csv", index=False)
    result = {
        "n_documents": int(len(documents)),
        "n_blocks": int(len(blocks)),
        "n_queries": int(len(query_ids)),
        "n_original_queries": int((queries["source"] == "cisi").sum()),
        "n_synthetic_full_queries": int((queries["source"] == "synthetic_full").sum()),
        "n_original_scored_queries": int(
            sum(query_source_by_id[qid] == "cisi" for qid in query_ids)
        ),
        "n_synthetic_scored_queries": int(
            sum(query_source_by_id[qid] == "synthetic_full" for qid in query_ids)
        ),
        "n_val_queries": int(len(val_query_ids)),
        "n_test_queries": int(len(test_query_ids)),
        "alpha": alpha,
        "fallback_threshold": fallback_threshold,
        "overall": overall.to_dict(orient="records"),
        "by_overlap_group": by_group.to_dict(orient="records"),
    }
    (artifacts_dir / "case_study_results.json").write_text(json.dumps(result, indent=2))

    return {
        "documents": documents,
        "queries": queries,
        "qrels": qrels,
        "blocks": blocks,
        "pairs": pairs,
        "overall": overall,
        "by_group": by_group,
        "alpha": alpha,
        "fallback_threshold": fallback_threshold,
        "val_query_ids": val_query_ids,
        "test_query_ids": test_query_ids,
        "result": result,
    }
