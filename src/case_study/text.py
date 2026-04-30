import re

import spacy
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.case_study.config import TOKEN_RE

_NLP = None


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def sentence_splitter():
    global _NLP
    if _NLP is None:
        _NLP = spacy.blank("en")
        _NLP.add_pipe("sentencizer")
    return _NLP


def split_sentences(text: str) -> list[str]:
    doc = sentence_splitter()(clean_text(text))
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall(str(text).lower())
        if token not in ENGLISH_STOP_WORDS
    }


def token_overlap(query: str, text: str) -> float:
    query_tokens = content_tokens(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & content_tokens(text)) / len(query_tokens)


def synthetic_query_from_block(text: str, max_terms: int = 10) -> str:
    tokens = []
    for token in TOKEN_RE.findall(str(text).lower()):
        if token not in ENGLISH_STOP_WORDS and token not in tokens:
            tokens.append(token)
        if len(tokens) == max_terms:
            break
    return " ".join(tokens)
