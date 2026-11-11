"""Task 2 - exploration and basic statistics."""
import re
from collections import Counter

import numpy as np
import pandas as pd

from src import config
from src.tokenizers_compare import regex_tokenize

WORD_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)*")   # alphabetic/numeric words only (no punctuation)


def words(text):
    return [w.lower() for w in WORD_RE.findall(text)]


def basic_stats(df: pd.DataFrame) -> dict:
    n_words = df["text"].str.split().str.len()
    by = df.assign(n_words=n_words).groupby("label")["n_words"].mean()
    counts = df["label"].value_counts()
    return {
        "n_docs": int(len(df)),
        "n_ham": int(counts.get("ham", 0)),
        "n_spam": int(counts.get("spam", 0)),
        "spam_pct": round(100 * counts.get("spam", 0) / len(df), 2),
        "n_duplicates": int(df.duplicated(subset=["label", "text"]).sum()),
        "avg_words": round(float(n_words.mean()), 2),
        "median_words": float(n_words.median()),
        "min_words": int(n_words.min()),
        "max_words": int(n_words.max()),
        "avg_words_ham": round(float(by.get("ham", np.nan)), 2),
        "avg_words_spam": round(float(by.get("spam", np.nan)), 2),
        "avg_chars": round(float(df["text"].str.len().mean()), 1),
    }


def vocab_stats(texts) -> dict:
    counter = Counter()
    for t in texts:
        counter.update(words(t))
    tokens = sum(counter.values())
    types = len(counter)
    hapax = sum(1 for c in counter.values() if c == 1)
    freqs = np.array(sorted(counter.values(), reverse=True), dtype=float)
    top = min(1000, len(freqs))
    slope = float(np.polyfit(np.log(np.arange(1, top + 1)), np.log(freqs[:top]), 1)[0]) if top > 2 else float("nan")
    return {
        "n_docs": len(texts), "tokens": int(tokens), "types": int(types),
        "type_token_ratio": round(types / tokens, 4),
        "hapax": int(hapax), "hapax_pct_of_types": round(100 * hapax / types, 1),
        "zipf_slope": round(slope, 2),
        "top_words": counter.most_common(15),
    }, counter


def zipf_plot(counter: Counter, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    freqs = np.array(sorted(counter.values(), reverse=True), dtype=float)
    ranks = np.arange(1, len(freqs) + 1)
    plt.figure(figsize=(5, 3.4))
    plt.loglog(ranks, freqs, ".", markersize=3, label="observed")
    plt.loglog(ranks, freqs[0] / ranks, "-", linewidth=1, label="ideal Zipf (1/rank)")
    plt.xlabel("rank (log)"); plt.ylabel("frequency (log)"); plt.title("Zipf plot - SMS corpus")
    plt.legend(); plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def class_and_length_plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(7.5, 3.2))
    counts = df["label"].value_counts()
    ax[0].bar(counts.index, counts.values, color=["#4C78A8", "#E45756"][: len(counts)])
    for i, v in enumerate(counts.values):
        ax[0].text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax[0].set_title("Target distribution")
    n_words = df["text"].str.split().str.len()
    for lab, col in (("ham", "#4C78A8"), ("spam", "#E45756")):
        ax[1].hist(n_words[df["label"] == lab], bins=30, alpha=0.6, label=lab, color=col)
    ax[1].set_title("Message length (words)"); ax[1].set_xlabel("words"); ax[1].legend()
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


def vocab_steps_plot(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 3.2))
    names = [r["step"] for r in rows]
    vals = [r["vocab"] for r in rows]
    plt.bar(range(len(vals)), vals, color="#59A14F")
    plt.xticks(range(len(vals)), names, rotation=40, ha="right", fontsize=7)
    plt.ylabel("vocabulary size"); plt.title("Vocabulary size after each cleaning step")
    plt.tight_layout(); plt.savefig(path, dpi=160); plt.close()


MESSY_CHECKS = [
    ("HTML entities", re.compile(r"&\w+;|&#\d+;")),
    ("Non-ASCII / encoding noise", re.compile(r"[^\x00-\x7F]")),
    ("Irregular spacing", re.compile(r"\s{2,}|\s[,.!?]")),
    ("Stretched letters", re.compile(r"([^\W\d_])\1{3,}")),
    ("Shouting (ALL CAPS)", re.compile(r"(?:\b[A-Z]{3,}\b\W+){3,}")),
    ("Missing space after punctuation", re.compile(r"[a-z][.,!?][A-Za-z]{2,}")),
    ("Phone / URL / price noise", re.compile(r"(?i)www\.|\d{9,}|[£$]\d")),
    ("Mixed case in one word", re.compile(r"\b[a-z]+[A-Z][a-zA-Z]*\b")),
]


def find_messy_examples(texts, seed=config.RANDOM_SEED, max_examples=5):
    """One real example for each of up to `max_examples` different kinds of mess."""
    rng = np.random.RandomState(seed)
    found = []
    for kind, pat in MESSY_CHECKS:
        pool = [t for t in texts if pat.search(t) and len(t) <= 200]
        if pool:
            found.append({"kind": kind, "example": pool[rng.randint(len(pool))], "n_messages": sum(1 for t in texts if pat.search(t))})
    return found[:max_examples]
