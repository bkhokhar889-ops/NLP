"""Download the NLTK resources used by the project (run once after installing requirements)."""
import nltk

RESOURCES = ["stopwords", "wordnet", "omw-1.4", "punkt", "punkt_tab"]

if __name__ == "__main__":
    for r in RESOURCES:
        nltk.download(r, quiet=False)
    print("NLTK resources ready.")
