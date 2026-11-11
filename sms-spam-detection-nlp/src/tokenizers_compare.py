"""Task 3 - compare tokenizers on SMS text.

Four tokenizers: whitespace split (baseline), a custom regex tokenizer, NLTK word_tokenize and
the spaCy English rule-based tokenizer (spacy.blank('en'), no model download needed).
"""
import re

_REGEX_TOK = re.compile(r"\w+(?:'\w+)*|[^\w\s]")
_SPACY_NLP = None


def whitespace_tokenize(text):
    return text.split()


def regex_tokenize(text):
    """Words (keeping inner apostrophes, so "don't" stays one token) or single symbols."""
    return _REGEX_TOK.findall(text)


def nltk_tokenize(text):
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)


def spacy_tokenize(text):
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        _SPACY_NLP = spacy.blank("en")
    return [t.text for t in _SPACY_NLP.tokenizer(text)]


def available_tokenizers():
    toks = {"whitespace": whitespace_tokenize, "regex": regex_tokenize, "nltk": nltk_tokenize}
    try:
        import spacy  # noqa: F401
        toks["spacy"] = spacy_tokenize
    except ImportError:
        print("  (spaCy not installed - skipping spaCy tokenizer)")
    return toks


# Indicators of tokenization challenges, counted over the corpus
CHALLENGES = {
    "contractions (don't, i'm)": re.compile(r"\b\w+'\w+\b"),
    "letters+digits slang (2day, gr8, 4u)": re.compile(r"\b(?=\w*\d)(?=\w*[A-Za-z])\w+\b"),
    "URLs / domains": re.compile(r"(?i)www\.|https?://|\.com\b|\.co\.uk\b"),
    "emoticons (:) ;-) <3)": re.compile(r"(?<!\S)(?:[:;=]['-]?[)(DPpOo|/\\]+|<3)(?!\w)"),
    "hyphenated words (e-mail, x-mas)": re.compile(r"\b[A-Za-z]+-[A-Za-z]+\b"),
    "slash-joined words (and/or, 10p/min)": re.compile(r"\w/\w"),
    "missing space after punctuation (ok.see)": re.compile(r"[a-z][.,!?][A-Za-z]{2,}"),
    "non-ASCII symbols (£, accents, mojibake)": re.compile(r"[^\x00-\x7F]"),
    "ellipsis / repeated dots": re.compile(r"\.{2,}"),
}


def challenge_counts(texts):
    n = len(texts)
    out = []
    for name, pat in CHALLENGES.items():
        c = sum(1 for t in texts if pat.search(t))
        out.append({"challenge": name, "n_messages": c, "pct": round(100 * c / n, 1)})
    return out


def pick_demo_sentences(texts, seed=42, k=5):
    """Pick short real messages that show different tokenization problems."""
    import random
    rng = random.Random(seed)
    wanted = ["contractions (don't, i'm)", "URLs / domains", "emoticons (:) ;-) <3)",
              "letters+digits slang (2day, gr8, 4u)", "non-ASCII symbols (£, accents, mojibake)"]
    chosen = []
    for name in wanted:
        pat = CHALLENGES[name]
        pool = [t for t in texts if pat.search(t) and 25 <= len(t) <= 110 and t not in chosen]
        if pool:
            chosen.append(rng.choice(pool))
    return chosen[:k]


def compare_on_sentences(sentences):
    toks = available_tokenizers()
    return [{"text": s, **{name: fn(s) for name, fn in toks.items()}} for s in sentences]


def compare_on_corpus(texts):
    toks = available_tokenizers()
    rows = []
    for name, fn in toks.items():
        all_tokens = [fn(t) for t in texts]
        vocab = {tok.lower() for ts in all_tokens for tok in ts}
        rows.append({"tokenizer": name,
                     "avg_tokens_per_msg": round(sum(map(len, all_tokens)) / len(texts), 2),
                     "vocab_size_lowercased": len(vocab)})
    return rows
