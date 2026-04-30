# Semantic Search Engine

## What this is

CISI experiment with TF-IDF, Sentence-BERT, and a few simple hybrids.
Search is done on text blocks. Document scores come from block scores.

## Run

Run commands from the project root.

Install:

```bash
pip install -r requirements.txt
```

Run the experiment:

```bash
python run.py
```

Notebook:

```bash
jupyter notebook semantic_search_case_study.ipynb
```

## Files

- `run.py` — runs the experiment
- `src/search_case_study.py` — short import wrapper
- `src/case_study/data.py` — blocks, qrels, synthetic full-overlap queries
- `src/case_study/tfidf.py` — TF-IDF
- `src/case_study/embeddings.py` — saved embedding loader
- `src/case_study/hybrids.py` — weighted and fallback hybrids
- `src/case_study/ranking.py` — rankings and interleaving hybrid
- `src/case_study/metrics.py` — MRR / hit@10
- `src/case_study/tuning.py` — alpha and threshold tuning
- `src/case_study/pipeline.py` — main experiment code

Embeddings are already saved in `artifacts/`.

## Current numbers

CISI has almost no exact-overlap pairs. I add 80 short full-overlap queries
from document blocks. `Hybrid interleave` alternates TF-IDF and BERT rankings
and skips docs that were already selected.

| Method | Block MRR | Doc MRR | Doc Hit@10 |
|---|---:|---:|---:|
| Hybrid interleave | 0.7957 | 0.8058 | 0.9744 |
| BERT | 0.7590 | 0.7952 | 0.9487 |
| Hybrid weighted | 0.7811 | 0.7897 | 0.9573 |
| TF-IDF | 0.7720 | 0.7733 | 0.9231 |
| Hybrid fallback | 0.7720 | 0.7733 | 0.9231 |

## Contacts

This case study was developed by Danis Sharafiev. 
Innopolis mail: d.sharafiev@innopolis.university
