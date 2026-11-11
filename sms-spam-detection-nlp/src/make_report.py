"""Generate the Phase-1 PDF report from reports/results.json + team.json (+ screenshots).

    python -m src.make_report              # -> reports/Assignment1_Report.pdf

All numbers come from results.json (written by src.run_pipeline), so the report always matches
the code. Put your terminal/notebook screenshots (png/jpg) in reports/screenshots/ and they are
appended as an appendix.
"""
import json
import sys
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from src import config

OUT = config.REPORTS / "Assignment1_Report.pdf"
W = A4[0] - 3.4 * cm   # usable width

ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=8.6, leading=11, alignment=TA_JUSTIFY, spaceAfter=3)
H1 = ParagraphStyle("h1", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, spaceBefore=7, spaceAfter=3, textColor=colors.HexColor("#1F3A5F"))
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=15, leading=18, spaceAfter=2)
SUB = ParagraphStyle("sub", parent=BODY, alignment=1, fontSize=9.5, textColor=colors.HexColor("#444444"))
CELL = ParagraphStyle("cell", parent=BODY, fontSize=7.3, leading=8.8, alignment=0, spaceAfter=0)
MONO = ParagraphStyle("mono", parent=CELL, fontName="Courier", fontSize=6.2, leading=7.4)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=7.5, textColor=colors.HexColor("#555555"), alignment=0)


def t(x):
    """Escape text and keep only characters the built-in PDF fonts can draw."""
    return escape(str(x).encode("cp1252", "replace").decode("cp1252"))


def P(x, style=BODY):
    return Paragraph(x, style)


def C(x, style=CELL):
    return Paragraph(t(x), style)


def table(rows, widths, header=True, mono_cols=()):
    data = []
    for r_i, row in enumerate(rows):
        cells = []
        for c_i, cell in enumerate(row):
            if isinstance(cell, Paragraph):
                cells.append(cell)
            elif r_i == 0 and header:
                cells.append(Paragraph(f"<b>{t(cell)}</b>", CELL))
            else:
                cells.append(C(cell, MONO if c_i in mono_cols else CELL))
        data.append(cells)
    tb = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B8C2CC")), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E6EDF5")))
    tb.setStyle(TableStyle(style))
    return tb


def img(path, width):
    if not path.exists():
        return Spacer(1, 1)
    w, h = ImageReader(str(path)).getSize()
    return Image(str(path), width=width, height=width * h / w)


def toks(lst, n=26):
    s = " | ".join(map(str, lst[:n]))
    return s + (" ..." if len(lst) > n else "")


def build_main(R, team):
    S = R["stats"]; vs = R["vocab_sample"]; vf = R["vocab_full"]; CL = R["cleaning"]; O = R["output"]; M = R["dataset_meta"]
    st = []
    # ---------------- title block
    st += [P("SMS Spam Detection - Project Phase 1", TITLE),
           P("Data Acquisition, Text Processing &amp; Cleaning &nbsp;|&nbsp; " + t(team.get("course", "Introduction to NLP")) + " &nbsp;|&nbsp; Assignment 1", SUB),
           P(f"<b>Team:</b> {t(team['team_name'])}", SUB), Spacer(1, 3)]
    mem = [["Member", "Roll number"]] + [[m["name"], m["roll_no"]] for m in team["members"]]
    st += [table(mem, [W * 0.65, W * 0.35]), Spacer(1, 2)]

    # ---------------- 1 topic & dataset
    st.append(P("1. Topic &amp; dataset", H1))
    st.append(P(
        "<b>Topic.</b> SMS spam detection: decide whether a short text message is <i>spam</i> (unwanted promotion, prize "
        "scam, premium-rate service) or <i>ham</i> (a normal personal message). <b>Why this topic.</b> SMS text is short, noisy "
        "and informal (slang, missing spaces, stretched letters, phone numbers, prices, emoticons), so every cleaning decision "
        "of this assignment has a visible effect; the label is binary and reliable, the corpus is small enough to run from scratch "
        "in seconds, and it is a classic benchmark, so Phase 2 (classification) has clear baselines to compare with."))
    ds = [["Item", "Details"],
          ["Dataset", M["name"]], ["Source / link", M["page"]],
          ["Size", f"{S['n_docs']} messages ({S['n_ham']} ham, {S['n_spam']} spam = {S['spam_pct']}% spam)"],
          ["Licence", M["licence"]], ["Language", "Mainly English (UK/Singapore SMS), informal style"],
          ["Future target (Phase 2)", "label: spam = 1 / ham = 0 (binary text classification)"],
          ["Approved by", team.get("topic_approved_by", "")], ["Citation", M["citation"]]]
    st.append(table(ds, [W * 0.22, W * 0.78]))

    # ---------------- 2 exploration
    st.append(P("2. Exploration", H1))
    ex = [["Statistic", "Value"],
          ["Number of documents", S["n_docs"]],
          ["Average length (words) - all / ham / spam", f"{S['avg_words']} / {S['avg_words_ham']} / {S['avg_words_spam']}  (median {S['median_words']}, min {S['min_words']}, max {S['max_words']})"],
          ["Target distribution", f"ham {S['n_ham']} ({100 - S['spam_pct']:.2f}%), spam {S['n_spam']} ({S['spam_pct']}%) - imbalanced"],
          ["Exact duplicate messages", f"{S['n_duplicates']} (kept, flagged in the output column is_duplicate)"],
          [f"Sample of {vs['n_docs']} msgs: tokens / types / TTR", f"{vs['tokens']} / {vs['types']} / {vs['type_token_ratio']}"],
          ["Full corpus: tokens / types / TTR", f"{vf['tokens']} / {vf['types']} / {vf['type_token_ratio']}"],
          ["Words occurring once (hapax)", f"{vf['hapax']} = {vf['hapax_pct_of_types']}% of all types"],
          ["Most frequent words", ", ".join(f"{w} ({c})" for w, c in vf["top_words"][:10])]]
    st.append(table(ex, [W * 0.36, W * 0.64]))
    st.append(Spacer(1, 3))
    st.append(Table([[img(config.FIGURES / "class_length.png", W * 0.58), img(config.FIGURES / "zipf.png", W * 0.40)]], colWidths=[W * 0.6, W * 0.4]))
    st.append(P(
        f"<b>Zipf / long tail.</b> Frequency falls steeply with rank: a handful of function words and spam words ({t(', '.join(w for w, _ in vf['top_words'][:5]))}) "
        f"cover a large share of all {vf['tokens']} tokens, while {vf['hapax_pct_of_types']}% of the {vf['types']} types appear only once "
        f"(the long tail: typos, slang, numbers, names). The fitted log-log slope for the top ranks is {vf['zipf_slope']} (a perfect Zipf law is about -1). "
        "A long tail like this means a huge sparse vocabulary - the main reason to clean and normalise before modelling."))
    mz = [["Kind of mess", "Real example from the dataset"]] + [[f"{m['kind']} ({m['n_messages']} msgs)", m["example"]] for m in R["messy_examples"]]
    st.append(table(mz, [W * 0.27, W * 0.73]))

    # ---------------- 3 tokenization & regex
    st.append(P("3. Tokenization &amp; regular expressions", H1))
    st.append(P("<b>Methods compared:</b> (1) whitespace split (baseline), (2) our regex tokenizer <font face='Courier'>\\w+(?:'\\w+)*|[^\\w\\s]</font> "
                "(words with inner apostrophes, or one symbol), (3) NLTK <font face='Courier'>word_tokenize</font>, (4) spaCy's rule-based English tokenizer (<font face='Courier'>spacy.blank('en')</font>)."))
    names = [k for k in R["tokenizer_demo"][0] if k != "text"] if R["tokenizer_demo"] else []
    for d in R["tokenizer_demo"][:3]:
        rows = [["Tokenizer", "Tokens"]] + [[n, toks(d[n])] for n in names]
        st.append(KeepTogether([P(f"<b>Message:</b> {t(d['text'])}", CAP), table(rows, [W * 0.13, W * 0.87]), Spacer(1, 3)]))
    cc = [["Tokenizer", "Avg tokens / message", "Vocabulary (lowercased types)"]] + [[r["tokenizer"], r["avg_tokens_per_msg"], r["vocab_size_lowercased"]] for r in R["tokenizer_corpus"]]
    st.append(table(cc, [W * 0.25, W * 0.35, W * 0.40]))
    st.append(Spacer(1, 3))
    rg = [["Pattern", "Regular expression", "What it does and why it matters here", "Msgs"]]
    for r in R["regex"]:
        rg.append([r["name"], r["pattern"], r["explanation"], f"{r['n_messages']}" + (f" e.g. {r['examples'][0]}" if r["examples"] else "")])
    st.append(table(rg, [W * 0.11, W * 0.31, W * 0.46, W * 0.12], mono_cols=(1,)))
    ch = {c["challenge"]: c for c in R["tokenizer_challenges"]}
    def pc(k):
        c = next((v for kk, v in ch.items() if kk.startswith(k)), None)
        return f"{c['n_messages']} msgs, {c['pct']}%" if c else "n/a"
    st.append(P("<b>Tokenization challenges found in this data.</b>"))
    st.append(P(f"<b>(a) Contractions and negations</b> ({pc('contractions')}). Our regex keeps <i>don't</i> as one token while NLTK/spaCy split it into <i>do</i> + <i>n't</i>; "
                "either way the negation must not be lost, so the pipeline expands contractions (<i>don't</i> -> <i>do not</i>) and protects <i>not</i> from stop-word removal. "
                f"<b>(b) SMS slang with digits</b> such as <i>2day, gr8, 4u</i> ({pc('letters+digits')}): a number rule would wrongly cut them, so numbers are replaced only when they stand alone. "
                f"<b>(c) Web addresses, phone numbers, prices and emoticons</b> (URLs {pc('URLs')}, emoticons {pc('emoticons')}) contain dots, slashes, digits and punctuation, "
                "so a punctuation-based tokenizer shreds them into meaningless pieces (e.g. <i>www . site . com</i>); they are matched with regexes first and become single class tokens. "
                f"<b>(d) Missing spaces / non-ASCII noise</b> (missing space after punctuation {pc('missing space')}; non-ASCII {pc('non-ASCII')}) glue words together or leave encoding debris, which tokenizers cannot repair."))

    # ---------------- 4 cleaning
    st.append(P("4. Cleaning pipeline", H1))
    st.append(P("All steps live in <font face='Courier'>src/cleaning.py</font> as one ordered list and are applied by the single function "
                "<font face='Courier'>clean_text()</font>. Vocabulary = number of distinct word tokens; the raw count is case-sensitive, so it is larger than the lowercased "
                "type count in Section 2."))
    cr = [["#", "Step (in order)", "Justification for SMS spam", "Vocab after"]]
    cr.append(["0", "raw text", "Starting point (no cleaning).", CL["steps"][0]["vocab"]])
    for i, s in enumerate(CL["steps"][1:], 1):
        cr.append([i, s["step"], s["justification"], s["vocab"]])
    st.append(table(cr, [W * 0.04, W * 0.19, W * 0.67, W * 0.10]))
    st.append(P(f"<b>Vocabulary before cleaning: {CL['vocab_before']} &nbsp;&rarr;&nbsp; after cleaning: {CL['vocab_after']} "
                f"(reduction {CL['reduction_pct']}%).</b> Order logic: Unicode and HTML normalisation first (so regexes see consistent characters); entity replacement "
                "before lowercasing and punctuation removal (it needs the original punctuation and case); case signals measured before lowercasing; contractions expanded before "
                "punctuation removal (apostrophes still exist); stop words removed after lowercasing; lemmatisation last (needs clean lowercase words).", BODY))
    st.append(img(config.FIGURES / "vocab_by_step.png", W * 0.62))
    kept = ", ".join(CL["stopwords_kept"])
    sw = [["Stop word", "spam msgs", "ham msgs", "spam rate / ham rate", "kept?"]] + [
        [r["word"], r["spam_msgs"], r["ham_msgs"], f"{r['spam_rate']} / {r['ham_rate']}", "yes" if r["kept"] else "no (removed)"] for r in CL["stopword_check"][:6]]
    st.append(P(f"<b>Case folding and stop words.</b> We lowercase, but first save <i>caps_ratio</i>, <i>n_caps_words</i> and <i>n_exclaim</i> because shouting is a spam cue. "
                f"Stop-word removal uses the NLTK English list <i>minus</i> these words, which are kept in the text: {t(kept)}. Check - stop words whose usage differs most between spam and ham:"))
    if len(sw) > 1:
        st.append(table(sw, [W * 0.2, W * 0.15, W * 0.15, W * 0.30, W * 0.20]))
    sl = [["Word", "Porter stem", "WordNet lemma"]] + [[r["word"], r["stem"], r["lemma"]] for r in CL["stem_vs_lemma"][:8]]
    st.append(Spacer(1, 3))
    st.append(P(f"<b>Stemming vs lemmatization - we chose lemmatization.</b> Examples of the two outputs on frequent words in the corpus:"))
    st.append(table(sl, [W * 0.3, W * 0.3, W * 0.4]))
    st.append(P(f"Among the {CL['n_compared']} most frequent words, {CL['non_word_stems']} Porter stems and {CL['non_word_lemmas']} lemmas are not dictionary words in WordNet "
                "(slang such as <i>txt</i> or <i>u</i> fails in both). Stemming cuts words into fragments that are unreadable and can merge unrelated words "
                "(e.g. <i>customer</i> and <i>custom</i>); lemmatisation keeps real words, which makes error analysis in Phase 2 easier, and the corpus is small enough that its extra cost is negligible. "
                "A limitation is that WordNet only knows standard English, so SMS slang is left unchanged."))
    ba = [["Raw", "Cleaned"]] + [[r["raw"], r["clean"]] for r in CL["before_after"][:3]]
    st.append(table(ba, [W * 0.5, W * 0.5]))

    # ---------------- 5 reproducibility
    st.append(P("5. Reproducibility", H1))
    st.append(P("Fresh clone -> install -> run:"))
    st.append(P("<font face='Courier' size='7.5'>git clone " + t(team["repo_url"]) + " &amp;&amp; cd sms-spam-detection-nlp<br/>"
                "python -m venv .venv &amp;&amp; source .venv/bin/activate &nbsp;(Windows: .venv\\Scripts\\activate)<br/>"
                "pip install -r requirements.txt &amp;&amp; python -m src.setup_nltk<br/>"
                "python -m src.run_pipeline &nbsp;&nbsp;# downloads data, runs Tasks 1-5, writes " + t(O["path"]) + "</font>", BODY))
    st.append(P(f"Tagged commit: <b>{t(team['tag'])}</b> (commit {t(team['commit_hash'])}) at {t(team['repo_url'])}. "
                f"The output has {O['n_rows']} rows and the columns {t(', '.join(O['columns']))}; its SHA-256 is "
                f"<font face='Courier' size='6.5'>{O['sha256']}</font>, identical on every run/machine. "
                f"{O['n_empty_after_cleaning']} messages become empty after cleaning and {O['n_duplicates_flagged']} duplicates are flagged (not dropped) so Phase 2 can decide. "
                f"Fresh clone verified by: {t(team.get('fresh_clone_verified_by', ''))}."))

    # ---------------- 6 contribution
    st.append(P("6. Contribution statement", H1))
    ct = [["Member", "Roll no.", "Contribution"]] + [[m["name"], m["roll_no"], m["contribution"]] for m in team["members"]]
    st.append(table(ct, [W * 0.22, W * 0.12, W * 0.66]))
    return st


def screenshots_story():
    files = sorted(p for p in config.SCREENSHOTS.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not files:
        return []
    st = [PageBreak(), P("Appendix - Screenshots (proof that every task was run)", H1)]
    for p in files:
        w, h = ImageReader(str(p)).getSize()
        width = min(W * 0.92, 12.5 * cm * w / h) if h > w else W * 0.92
        st += [KeepTogether([img(p, width), P(t(p.name), CAP), Spacer(1, 4)])]
    return st


def main():
    if not config.RESULTS_JSON.exists():
        sys.exit("reports/results.json not found - run  python -m src.run_pipeline  first.")
    R = json.loads(config.RESULTS_JSON.read_text(encoding="utf-8"))
    team = json.loads(config.TEAM_JSON.read_text(encoding="utf-8"))
    placeholders = [k for k in ("team_name", "repo_url", "commit_hash", "topic_approved_by", "fresh_clone_verified_by")
                    if any(x in str(team.get(k, "")) for x in ("HERE", "YOUR-", "PASTE", "Instructor name", "Member name"))]
    placeholders += ["members" for m in team["members"] if "Member" in m["name"] or "ROLL-" in m["roll_no"]][:1]
    if placeholders:
        print("WARNING - edit team.json, these fields still hold placeholders:", ", ".join(placeholders))

    def doc(path):
        return SimpleDocTemplate(str(path), pagesize=A4, leftMargin=1.7 * cm, rightMargin=1.7 * cm, topMargin=1.4 * cm, bottomMargin=1.4 * cm,
                                 title="SMS Spam Detection - Phase 1", author=team["team_name"])
    d = doc(OUT)
    d.build(build_main(R, team) + screenshots_story())
    probe = doc(config.REPORTS / "_probe.pdf")
    probe.build(build_main(R, team))
    pages = probe.page
    (config.REPORTS / "_probe.pdf").unlink()
    print(f"Report written to {OUT}")
    print(f"Main report length (without screenshots appendix): {pages} pages", "- OK (target 4-6)" if 4 <= pages <= 6 else "- NOTE: the assignment asks for 4-6 pages")


if __name__ == "__main__":
    main()
