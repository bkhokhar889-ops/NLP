# SMS Spam Detection - NLP Project, Phase 1

**Course:** Introduction to NLP  |  **Assignment 1:** Data acquisition, text processing & cleaning
**Team:** see `team.json`  |  **Tag for submission:** `assignment-1`

We classify SMS messages as **spam** or **ham** (normal). Phase 1 (this repository) turns the raw, noisy
messages into a clean, documented, analysis-ready corpus. Phase 2 will train classifiers on the cleaned file.

---

## 1. Dataset (Task 1)

| Item | Details |
|---|---|
| **Name** | SMS Spam Collection |
| **Source** | UCI Machine Learning Repository - https://archive.ics.uci.edu/dataset/228/sms+spam+collection |
| **Direct download** | https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip |
| **Size** | about 5,574 messages (about 4,827 ham / 747 spam, ~13% spam). The exact numbers are printed by `src.run_pipeline` and stored in `reports/results.json`. |
| **Licence** | CC BY 4.0 (verify on the UCI page above) |
| **Language** | Mainly English, informal SMS style |
| **Prediction target (Phase 2)** | `label` - `spam` (1) or `ham` (0): binary text classification |
| **Citation** | Almeida, T.A., Gomez Hidalgo, J.M., Yamakami, A. (2011). *Contributions to the study of SMS spam filtering: new collection and results.* ACM DocEng 2011. |

The raw data is **not committed**. `src/load_data.py` downloads it into `data/raw/` automatically.
If your network blocks the download, download the zip manually, unzip it and put the file
`SMSSpamCollection` in `data/raw/` (or run `python -m src.run_pipeline --data path/to/file`).

## 2. Repository structure

```
sms-spam-detection-nlp/
|-- README.md                  <- this file
|-- requirements.txt           <- exact Python dependencies
|-- team.json                  <- team name, members, roll numbers, contributions (used by the report)
|-- src/
|   |-- config.py              <- paths and constants
|   |-- load_data.py           <- Task 1: download / load raw data
|   |-- explore.py             <- Task 2: statistics, Zipf, messy examples, plots
|   |-- tokenizers_compare.py  <- Task 3: whitespace vs regex vs NLTK vs spaCy
|   |-- regex_patterns.py      <- Task 3: the regular expressions (explained)
|   |-- cleaning.py            <- Task 4: the single cleaning pipeline  (clean_text)
|   |-- run_pipeline.py        <- runs Tasks 1-5 end-to-end, saves outputs
|   |-- make_report.py         <- builds the PDF report from the real results
|   `-- setup_nltk.py          <- downloads NLTK resources
|-- notebooks/01_exploration_and_cleaning.ipynb   <- exploration notebook (Tasks 1-5, with outputs)
|-- tests/test_cleaning.py     <- unit tests for the pipeline (pytest)
|-- scripts/verify_fresh_clone.sh                 <- proves a fresh clone reproduces the output
|-- data/raw/                  <- downloaded raw data (git-ignored)
|-- data/processed/            <- sms_spam_clean.csv (Task 5 output, used by Phase 2)
`-- reports/                   <- figures/, screenshots/, results.json, Assignment1_Report.pdf
```

## 3. Setup (Python 3.11+)

```bash
git clone <REPO-URL> && cd sms-spam-detection-nlp
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell:  .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m src.setup_nltk             # one-off: NLTK stopwords, wordnet, punkt
```

spaCy is used only through `spacy.blank("en")` (its rule-based tokenizer), so **no spaCy model download is needed**.

## 4. Run

```bash
python -m src.run_pipeline           # Tasks 1-5, prints every result, writes the cleaned data
python -m src.make_report            # builds reports/Assignment1_Report.pdf (after editing team.json)
pytest -q                            # optional: unit tests
jupyter notebook notebooks/01_exploration_and_cleaning.ipynb   # interactive version
```

Outputs:

| File | Content |
|---|---|
| `data/processed/sms_spam_clean.csv` | `label, label_num, text_raw, text_clean, n_tokens_clean, caps_ratio, n_caps_words, n_exclaim, is_duplicate` |
| `reports/results.json` | every number used in the report |
| `reports/figures/*.png` | class distribution, length histogram, Zipf plot, vocabulary per step |

## 5. The cleaning pipeline in one glance (Task 4)

`unicode_nfc -> html_cleanup -> normalize_punctuation -> replace_entities (email, URL, emoticon, money, phone, short code, number)
-> lowercase -> expand_contractions -> collapse_repeats -> strip_punctuation -> remove_stopwords -> lemmatize`

* Capitalisation and `!` counts are measured **before** lowercasing and saved as columns (shouting is a spam signal).
* Stop-word removal keeps negations, `you/your`, `now`, `only`, `off`.
* Lemmatization (WordNet) is used instead of stemming; justification is in `src/cleaning.py` and the report.
* Everything is one function: `from src.cleaning import clean_text`.

## 6. Reproducibility check

`bash scripts/verify_fresh_clone.sh <REPO-URL>` clones the repository into a temp folder, installs it, runs the
pipeline and compares the SHA-256 of the cleaned CSV with your local one. The hash is also printed by `run_pipeline`.

## 7. Using the output in Phase 2

```python
import pandas as pd
df = pd.read_csv("data/processed/sms_spam_clean.csv")
df = df[df.is_duplicate == 0]        # drop exact duplicates BEFORE splitting train/test (avoids leakage)
X, y = df["text_clean"].fillna(""), df["label_num"]
```
Do not change the dataset or the cleaning between phases.

## 8. Git workflow

Work on feature branches, open pull requests, commit early and often, tag the submission:
`git tag -a assignment-1 -m "Phase 1 submission" && git push origin assignment-1`
