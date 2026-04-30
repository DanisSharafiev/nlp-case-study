import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.case_study.text import content_tokens


def fit_tfidf(blocks: pd.DataFrame) -> tuple[TfidfVectorizer, object]:
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words="english",
        lowercase=True,
    )
    matrix = vectorizer.fit_transform(blocks["text"].fillna(""))
    return vectorizer, normalize(matrix, norm="l2", copy=False)


def tfidf_query_scores(
    vectorizer: TfidfVectorizer,
    block_matrix,
    query_texts: list[str],
) -> np.ndarray:
    query_matrix = normalize(vectorizer.transform(query_texts), norm="l2", copy=False)
    return (query_matrix @ block_matrix.T).toarray()


def query_vocab_coverage(query: str, vectorizer: TfidfVectorizer) -> float:
    tokens = content_tokens(query)
    if not tokens:
        return 0.0
    vocab = vectorizer.vocabulary_
    return sum(token in vocab for token in tokens) / len(tokens)
