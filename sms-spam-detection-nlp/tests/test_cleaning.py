"""Unit tests. Run with:  pytest -q   (needs `python -m src.setup_nltk` first)."""
import pytest

pytest.importorskip("nltk")
from src import cleaning, regex_patterns as rp  # noqa: E402


def test_deterministic():
    s = "WINNER!! Claim £1000 now at www.prize.com :)"
    assert cleaning.clean_text(s) == cleaning.clean_text(s)


def test_entities_replaced():
    out = cleaning.clean_text("Call 08000839402 or visit www.claim.com to win £1000, text WIN to 80086")
    for tok in ("phonetoken", "urltoken", "moneytoken", "shortcodetoken"):
        assert tok in out
    assert "0800" not in out and "www" not in out


def test_negation_and_you_survive_stopwords():
    out = cleaning.clean_text("You don't have to pay, you can not lose")
    assert "not" in out.split() and "you" in out.split()


def test_html_marker_becomes_number_token():
    assert "numtoken" in cleaning.clean_text("call me in &lt;#&gt; mins")


def test_repeated_letters_collapsed():
    assert "soo" in cleaning.clean_text("I am sooooo happy").split()


def test_non_string_is_empty():
    assert cleaning.clean_text(None) == ""


def test_signals_before_lowercasing():
    sig = cleaning.extract_signals("FREE entry!!! WIN now")
    assert sig["n_caps_words"] == 2 and sig["n_exclaim"] == 3 and sig["caps_ratio"] > 0.4


def test_email_not_eaten_by_url_rule():
    assert "emailtoken" in cleaning.clean_text("mail bob@site.co.uk please")


def test_url_pattern_ignores_trailing_dot():
    assert rp.URL.search("go to www.site.com.").group(0) == "www.site.com"
