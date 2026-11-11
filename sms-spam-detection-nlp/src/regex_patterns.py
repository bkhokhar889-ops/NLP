"""Task 3 - regular expressions designed for SMS spam text.

Every pattern has: a name, the compiled regex, a plain-English explanation (used verbatim in
the report) and the placeholder token the cleaning pipeline replaces matches with.
"""
import re

# Placeholder tokens (plain letters so tokenizers/lemmatizers leave them untouched)
PLACEHOLDERS = {
    "url": "urltoken",
    "email": "emailtoken",
    "money": "moneytoken",
    "phone": "phonetoken",
    "shortcode": "shortcodetoken",
    "number": "numtoken",
    "emoticon": "emoticontoken",
}

URL = re.compile(
    r"(?i)\b(?:https?://|www\.)[^\s]*[^\s.,;:!?)\]'\"]"
    r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|net|org|co\.uk)\b(?:/[^\s]*[^\s.,;:!?])?"
)
EMAIL = re.compile(r"(?i)\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
MONEY = re.compile(
    r"(?i)(?:[£$€]\s?\d[\d,]*(?:\.\d+)?(?:\s?(?:k|m|million|thousand))?"
    r"|\b\d[\d,]*(?:\.\d+)?\s?(?:p|pence|pounds?|gbp|usd|dollars?|euros?|quid)\b(?!\.?\w))"
)
PHONE = re.compile(
    r"(?<![\w£$€.])(?:\+\d{1,3}[ -]?)?\(?\d{3,5}\)?[ -]?\d{3,4}[ -]?\d{3,4}(?![\w])"
)
SHORTCODE = re.compile(r"(?<![\w.£$€])\d{5,6}(?!\w|\.\d)")
NUMBER = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)")
EMOTICON = re.compile(r"(?<!\S)(?:[:;=]['-]?[)(DPpOo|/\\]+|<3)(?!\w)")
HTML_ENTITY = re.compile(r"(?i)&(?:#\d+|#x[0-9a-f]+|[a-z]+);")
REPEATED_CHAR = re.compile(r"(?!(?i:www))([^\W\d_])\1{2,}")  # ignores the www of web addresses
ALL_CAPS_WORD = re.compile(r"\b[A-Z]{2,}\b")

# name -> (pattern, explanation)  (used for the report table and the console demo)
PATTERNS = {
    "URL": (
        URL,
        "Matches http(s):// or www. links (up to the next space, excluding trailing punctuation) "
        "and bare domains ending in .com/.net/.org/.co.uk. Spam pushes links ('claim at www...'), "
        "so the link is replaced by one token instead of being split into meaningless pieces.",
    ),
    "UK/intl phone number": (
        PHONE,
        "A 3-5 digit block followed by two 3-4 digit blocks (spaces or hyphens optional), with an "
        "optional +country code, i.e. 9-13 digits in total. Spam asks people to 'call 0800...'; "
        "ham rarely contains a full phone number.",
    ),
    "Money amount": (
        MONEY,
        "A currency symbol (£ $ €) followed by an amount, or an amount followed by p / pence / "
        "pounds / dollars / quid. Prizes and charges ('£1000 cash', '150p/min') are strong spam "
        "cues, so the amount is replaced by 'moneytoken' (the exact value is noise).",
    ),
    "Premium short code": (
        SHORTCODE,
        "A standalone 5-6 digit number (e.g. 'text WIN to 80086'). These are premium-rate SMS "
        "short codes that appear mainly in spam, so they get their own token.",
    ),
    "Email address": (
        EMAIL,
        "local-part@domain.tld. Rare in this corpus but present in a few spam messages.",
    ),
    "Emoticon": (
        EMOTICON,
        "Western emoticons such as :) ;-) :D :( and <3, only when they start a whitespace-separated "
        "chunk. They are frequent in personal (ham) messages and would be destroyed by punctuation "
        "removal, so they are kept as 'emoticontoken'.",
    ),
    "HTML entity": (
        HTML_ENTITY,
        "Matches &lt; &gt; &amp; &#39; style entities left over from the web export of the "
        "dataset (the string '&lt;#&gt;' appears many times and marks an anonymised number).",
    ),
    "Repeated letters": (
        REPEATED_CHAR,
        "A letter repeated 3 or more times ('sooooo', 'yeeesss'). Collapsing to two letters keeps "
        "the emphasis but shrinks the number of word variants.",
    ),
    "ALL-CAPS word": (
        ALL_CAPS_WORD,
        "Two or more consecutive capital letters ('FREE', 'WINNER'). Used to count shouting, "
        "which is measured BEFORE lowercasing because spam uses capitals heavily.",
    ),
}


def demo_counts(texts, examples_per_pattern: int = 2):
    """For every pattern: number of messages with a match and a few real examples."""
    rows = []
    for name, (pat, expl) in PATTERNS.items():
        n_msgs, examples = 0, []
        for t in texts:
            m = pat.search(t)
            if m:
                n_msgs += 1
                if len(examples) < examples_per_pattern:
                    examples.append(m.group(0))
        rows.append(
            {"name": name, "pattern": pat.pattern, "explanation": expl,
             "n_messages": n_msgs, "examples": examples}
        )
    return rows
