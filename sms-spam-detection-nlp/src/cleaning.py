"""Task 4 - the single, reusable cleaning pipeline for SMS spam text.

Use it like this (identical output every time):

    from src.cleaning import clean_text
    clean_text("WINNER!! Claim £1000 now at www.prize.com")
    # -> 'winner claim moneytoken now urltoken'

The pipeline is an ORDERED list of steps (STEPS). Each step is (name, function, justification).
Order matters - the justification of the order is given in the comments and in the report.
"""
import html
import re
import unicodedata
from functools import lru_cache

from src import regex_patterns as rp

_TOKEN_RE = re.compile(r"\w+(?:'\w+)*")   # used only to MEASURE vocabulary size

# --------------------------------------------------------------------------------------
# Stop words: NLTK English list MINUS words our task needs
# --------------------------------------------------------------------------------------
# Negations flip meaning ("not free"), second-person words and urgency words are typical
# spam cues ("you have won", "call now", "only 10p"), so they must survive stop-word removal.
KEEP_WORDS = {
    "no", "nor", "not", "never", "cannot",
    "you", "your", "yours", "yourself", "yourselves",
    "now", "only", "off",
}


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset:
    """NLTK English stop words minus KEEP_WORDS (and minus every negation form)."""
    from nltk.corpus import stopwords
    base = set(stopwords.words("english"))
    negations = {w for w in base if w.endswith("n't")} | {"don", "didn", "doesn", "hadn", "hasn",
                 "haven", "isn", "wasn", "weren", "won", "wouldn", "shouldn", "couldn", "aren",
                 "mightn", "mustn", "needn", "shan"}
    return frozenset(base - KEEP_WORDS - negations)


@lru_cache(maxsize=1)
def _lemmatizer():
    from nltk.stem import WordNetLemmatizer
    return WordNetLemmatizer()


@lru_cache(maxsize=1)
def _stemmer():
    from nltk.stem import PorterStemmer
    return PorterStemmer()


# --------------------------------------------------------------------------------------
# Steps (string -> string)
# --------------------------------------------------------------------------------------
def unicode_nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def html_cleanup(s: str) -> str:
    s = html.unescape(s)                       # &lt; -> <, &amp; -> &, &#39; -> '
    s = s.replace("<#>", f" {rp.PLACEHOLDERS['number']} ")  # corpus marker for an anonymised number
    return re.sub(r"<[^>]{1,40}>", " ", s)    # drop any real HTML tags


_PUNCT_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'", "\u00b4": "'", "`": "'",
    "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2026": "...",
    "\u00a0": " ", "\u200b": "", "\ufeff": "",
}


def normalize_punctuation(s: str) -> str:
    return "".join(_PUNCT_MAP.get(ch, ch) for ch in s)


def replace_entities(s: str) -> str:
    """Replace emails, URLs, emoticons, money, phones, short codes and numbers by tokens.
    ORDER INSIDE THIS STEP MATTERS: email before URL (else 'x.co.uk' inside an address is
    eaten as a URL), emoticons before numbers (else '<3' loses its '3'), money/phone/short
    code before the generic number rule."""
    P = rp.PLACEHOLDERS
    for name, pat in (("email", rp.EMAIL), ("url", rp.URL), ("emoticon", rp.EMOTICON),
                      ("money", rp.MONEY), ("phone", rp.PHONE), ("shortcode", rp.SHORTCODE),
                      ("number", rp.NUMBER)):
        s = pat.sub(f" {P[name]} ", s)
    return s


def lowercase(s: str) -> str:
    return s.lower()


_CONTRACTIONS = [
    (re.compile(r"\bcan't\b"), "can not"), (re.compile(r"\bwon't\b"), "will not"),
    (re.compile(r"\bain't\b"), "is not"),
    (re.compile(r"n't\b"), " not"), (re.compile(r"'re\b"), " are"), (re.compile(r"'m\b"), " am"),
    (re.compile(r"'ll\b"), " will"), (re.compile(r"'ve\b"), " have"), (re.compile(r"'d\b"), " would"),
    (re.compile(r"'s\b"), ""),
]


def expand_contractions(s: str) -> str:
    for pat, repl in _CONTRACTIONS:
        s = pat.sub(repl, s)
    return s


def collapse_repeats(s: str) -> str:
    return rp.REPEATED_CHAR.sub(r"\1\1", s)


def strip_punctuation(s: str) -> str:
    s = s.replace("_", " ")
    s = re.sub(r"[^\w\s]", " ", s)             # any remaining punctuation / symbols (inc. £ and ')
    return re.sub(r"\s+", " ", s).strip()


def remove_stopwords(s: str) -> str:
    sw = get_stopwords()
    return " ".join(t for t in s.split() if t not in sw)


_PLACEHOLDER_SET = set(rp.PLACEHOLDERS.values())


@lru_cache(maxsize=50000)
def lemmatize_word(w: str) -> str:
    if w in _PLACEHOLDER_SET or w.isdigit() or any(c.isdigit() for c in w):
        return w
    lem = _lemmatizer()
    out = lem.lemmatize(w, "v")                 # verbs first ('calls','won' ...)
    return out if out != w else lem.lemmatize(w, "n")   # then nouns ('messages', 'prizes')


def lemmatize(s: str) -> str:
    return " ".join(lemmatize_word(t) for t in s.split())


# (name, function, one-line justification tied to SMS spam data)
STEPS = [
    ("unicode_nfc", unicode_nfc,
     "Applied first so visually identical characters (accents, composed symbols such as the pound sign) "
     "have ONE code-point form before any regex or tokenizer sees them."),
    ("html_cleanup", html_cleanup,
     "The corpus contains HTML entities (&lt; &gt; &amp;) from its web export; '&lt;#&gt;' is the "
     "dataset's marker for an anonymised number, so it becomes a number token."),
    ("normalize_punctuation", normalize_punctuation,
     "Curly quotes, long dashes, non-breaking spaces and zero-width characters differ between phones; "
     "mapping them to plain ASCII lets the later regexes and contraction rules match."),
    ("replace_entities", replace_entities,
     "URLs, e-mails, phone numbers, prices, 5-6 digit short codes, plain numbers and emoticons are "
     "replaced by class tokens: the exact value is noise (huge vocabulary) but the PRESENCE is a spam "
     "signal. Done before lowercasing/punctuation removal because it needs the original punctuation."),
    ("lowercase", lowercase,
     "Case folding merges 'FREE', 'Free', 'free'. Capitalisation IS a spam signal, so it is measured "
     "(caps ratio, caps-word count, '!' count) BEFORE this step and saved as numeric columns."),
    ("expand_contractions", expand_contractions,
     "Turns \"don't\" into 'do not' so negations are kept as separate 'not' tokens that survive stop-word "
     "removal; also handles 'u'll', 'i'm', 'it's'."),
    ("collapse_repeats", collapse_repeats,
     "'sooooo', 'yessss' -> 'soo', 'yess': keeps emphasis but removes thousands of spelling variants."),
    ("strip_punctuation", strip_punctuation,
     "Removes punctuation/symbols and collapses whitespace. After the entity step nothing useful "
     "depends on punctuation (the '!' count was recorded earlier)."),
    ("remove_stopwords", remove_stopwords,
     "NLTK English list minus negations, 'you/your', 'now', 'only', 'off': these carry meaning in "
     "spam ('not free', 'you have won', 'call now'). Done after lowercasing/contractions so the list matches."),
    ("lemmatize", lemmatize,
     "WordNet lemmatization (verb then noun). Chosen over stemming because it returns real words "
     "('messages'->'message') which keeps the text readable and avoids over-stemming "
     "('customer'->'custom'). Done last, on lowercase tokens, as WordNet expects."),
]
STEP_NAMES = [n for n, _, _ in STEPS]


def apply_steps(text: str, upto: str = None) -> str:
    """Apply the pipeline up to (and including) the step called `upto` (default: all)."""
    s = text
    for name, fn, _ in STEPS:
        s = fn(s)
        if name == upto:
            break
    return s


def clean_text(text) -> str:
    """THE cleaning function. Same input -> same output, always."""
    if not isinstance(text, str):
        return ""
    return apply_steps(text)


# --------------------------------------------------------------------------------------
# Signals measured BEFORE lowercasing (capitalisation / punctuation are useful for spam)
# --------------------------------------------------------------------------------------
def extract_signals(text: str) -> dict:
    s = normalize_punctuation(html_cleanup(unicodedata.normalize("NFC", text)))
    s = replace_entities(s)                     # so 'WWW.X.COM' is not counted as shouting
    letters = [c for c in s if c.isalpha()]
    upper = sum(c.isupper() for c in letters)
    return {
        "caps_ratio": round(upper / len(letters), 4) if letters else 0.0,
        "n_caps_words": len(rp.ALL_CAPS_WORD.findall(s)),
        "n_exclaim": s.count("!"),
    }


# --------------------------------------------------------------------------------------
# Helpers for the report
# --------------------------------------------------------------------------------------
def measure_tokens(s: str):
    return _TOKEN_RE.findall(s)


def vocab_size(texts) -> int:
    v = set()
    for t in texts:
        v.update(measure_tokens(t))
    return len(v)


def vocab_by_step(texts):
    """Vocabulary size of the whole corpus before cleaning and after every step."""
    texts = list(texts)
    rows = [{"step": "raw (no cleaning)", "vocab": vocab_size(texts)}]
    current = texts
    for name, fn, _ in STEPS:
        current = [fn(t) for t in current]
        rows.append({"step": name, "vocab": vocab_size(current)})
    return rows


def stem_word(w: str) -> str:
    return _stemmer().stem(w)
