"""Central configuration: paths, dataset facts and constants used by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
SCREENSHOTS = REPORTS / "screenshots"

RAW_FILE = DATA_RAW / "SMSSpamCollection"           # tab-separated: label <TAB> text
CLEAN_FILE = DATA_PROC / "sms_spam_clean.csv"        # the file Phase 2 will load
RESULTS_JSON = REPORTS / "results.json"              # numbers used by the report
TEAM_JSON = ROOT / "team.json"

# ---- Dataset facts (documented in README and report) ----
DATASET_NAME = "SMS Spam Collection (UCI Machine Learning Repository)"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATASET_LICENCE = "CC BY 4.0 (check the licence shown on the UCI dataset page)"
DATASET_CITATION = (
    "Almeida, T.A., Gomez Hidalgo, J.M., Yamakami, A. (2011). Contributions to the study of "
    "SMS spam filtering: new collection and results. ACM DocEng 2011."
)

RANDOM_SEED = 42
SAMPLE_SIZE = 1000   # sample used for the tokens-vs-types analysis in Task 2
