from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


artifacts = Path("artifacts")
overall = pd.read_csv(artifacts / "overall_metrics.csv")
overlap = pd.read_csv(artifacts / "overlap_metrics.csv")


def method_name(value):
    names = {
        "tfidf": "TF-IDF",
        "bert": "BERT",
        "hybrid_fallback": "Hybrid fallback",
        "hybrid_weighted": "Hybrid weighted",
        "hybrid_interleave": "Hybrid interleave",
    }
    return names[value]


def format_table(frame):
    frame = frame.copy()
    frame["method"] = frame["method"].map(method_name)
    for column in ["block_mrr", "doc_mrr", "doc_hit@10"]:
        frame[column] = frame[column].map(lambda value: f"{value:.4f}")
    return frame


def draw_table(ax, frame, columns, widths, font_size):
    ax.axis("off")
    table = ax.table(
        cellText=frame[columns].values,
        colLabels=columns,
        colWidths=widths,
        cellLoc="center",
        loc="upper left",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)

    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("#b9d9bd")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#008609")
            cell.get_text().set_color("white")
            cell.get_text().set_weight("bold")
        elif row % 2 == 0:
            cell.set_facecolor("#eff8f0")
        else:
            cell.set_facecolor("#fffdf8")
    return table


overall_table = format_table(overall).rename(
    columns={
        "method": "Method",
        "block_mrr": "Block MRR",
        "doc_mrr": "Doc MRR",
        "doc_hit@10": "Doc Hit@10",
    }
)

overlap_table = format_table(overlap).rename(
    columns={
        "overlap_group": "Group",
        "n_pairs": "Pairs",
        "n_queries": "Queries",
        "method": "Method",
        "block_mrr": "Block MRR",
        "doc_mrr": "Doc MRR",
        "doc_hit@10": "Doc Hit@10",
    }
)
overlap_table["Group"] = overlap_table["Group"].str.title()

fig = plt.figure(figsize=(8.27, 11.69), dpi=300)
fig.patch.set_facecolor("#f8f6f0")

fig.text(
    0.06,
    0.955,
    "Semantic Search Results",
    fontsize=24,
    fontweight="bold",
    color="#008609",
)
fig.text(
    0.06,
    0.925,
    "CISI dataset | 1460 documents | 1973 text blocks | 156 scored queries",
    fontsize=11,
    color="#4b5563",
)
fig.text(
    0.06,
    0.902,
    "MRR rewards a relevant result near the top. Hit@10 checks whether at least one relevant document appears in the top 10.",
    fontsize=9,
    color="#5b6470",
)

fig.text(0.06, 0.855, "Overall Results", fontsize=15, fontweight="bold", color="#1f2933")
overall_ax = fig.add_axes([0.06, 0.65, 0.88, 0.18])
draw_table(
    overall_ax,
    overall_table,
    ["Method", "Block MRR", "Doc MRR", "Doc Hit@10"],
    [0.43, 0.19, 0.19, 0.19],
    10,
)

fig.text(
    0.06,
    0.595,
    "Overlap Group Breakdown",
    fontsize=15,
    fontweight="bold",
    color="#1f2933",
)
fig.text(
    0.06,
    0.575,
    "Pairs are relevant query-document links. Queries are unique test queries inside each group.",
    fontsize=9,
    color="#5b6470",
)
overlap_ax = fig.add_axes([0.06, 0.11, 0.88, 0.445])
draw_table(
    overlap_ax,
    overlap_table,
    ["Group", "Pairs", "Queries", "Method", "Block MRR", "Doc MRR", "Doc Hit@10"],
    [0.12, 0.10, 0.10, 0.25, 0.14, 0.14, 0.15],
    8,
)

fig.text(
    0.06,
    0.055,
    "Best overall by Doc MRR and Hit@10: Hybrid interleave.",
    fontsize=11,
    fontweight="bold",
    color="#1f2933",
)

fig.savefig(artifacts / "results_tables_a4.png", bbox_inches="tight", facecolor=fig.get_facecolor())
