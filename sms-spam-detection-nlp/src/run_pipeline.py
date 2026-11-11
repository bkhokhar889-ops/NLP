"""Run the WHOLE of Phase 1 from a fresh clone:

    python -m src.run_pipeline

Steps (they match the assignment tasks and are printed as sections so that a screenshot of the
terminal proves every task was run):
    Task 1  download/load raw data
    Task 2  exploration + statistics (+ figures)
    Task 3  tokenizer comparison + regular expressions
    Task 4  cleaning pipeline, vocabulary before/after, stemming vs lemmatization
    Task 5  save the cleaned dataset (data/processed/sms_spam_clean.csv) + results.json for the report
"""
import argparse
import hashlib
import json
from collections import Counter

import pandas as pd

from src import cleaning, config, explore, load_data, regex_patterns
from src import tokenizers_compare as tc


def banner(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def sha256_of(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main(data_path=None):
    for d in (config.DATA_RAW, config.DATA_PROC, config.FIGURES):
        d.mkdir(parents=True, exist_ok=True)
    R = {"dataset_meta": {
        "name": config.DATASET_NAME, "page": config.DATASET_PAGE, "url": config.DATASET_URL,
        "licence": config.DATASET_LICENCE, "citation": config.DATASET_CITATION}}

    # ------------------------------------------------------------------ Task 1
    banner("TASK 1 - Dataset & environment: load raw data")
    df = load_data.load_raw(data_path)
    print(df.head(5).to_string())
    print(f"\nLoaded {len(df)} messages. Target = label (ham=0 / spam=1).")

    # ------------------------------------------------------------------ Task 2
    banner("TASK 2 - Exploration & basic statistics")
    stats = explore.basic_stats(df)
    print(json.dumps(stats, indent=2))
    sample = df.sample(n=min(config.SAMPLE_SIZE, len(df)), random_state=config.RANDOM_SEED)
    v_sample, _ = explore.vocab_stats(sample["text"].tolist())
    v_full, counter = explore.vocab_stats(df["text"].tolist())
    print(f"\nSample of {len(sample)} msgs: tokens={v_sample['tokens']}  types={v_sample['types']}  TTR={v_sample['type_token_ratio']}")
    print(f"Full corpus:  tokens={v_full['tokens']}  types={v_full['types']}  TTR={v_full['type_token_ratio']}")
    print(f"Hapax legomena (words seen once): {v_full['hapax']} = {v_full['hapax_pct_of_types']}% of types")
    print(f"Zipf log-log slope (top 1000 ranks): {v_full['zipf_slope']}  (ideal Zipf = -1)")
    print("Top 15 words:", v_full["top_words"])
    explore.zipf_plot(counter, config.FIGURES / "zipf.png")
    explore.class_and_length_plot(df, config.FIGURES / "class_length.png")
    messy = explore.find_messy_examples(df["text"].tolist())
    print("\nMessy examples:")
    for m in messy:
        print(f"  [{m['kind']}] ({m['n_messages']} msgs)  {m['example']!r}")
    R.update(stats=stats, vocab_sample=v_sample, vocab_full=v_full, messy_examples=messy)

    # ------------------------------------------------------------------ Task 3
    banner("TASK 3 - Tokenization & regular expressions")
    demo = tc.compare_on_sentences(tc.pick_demo_sentences(df["text"].tolist()))
    for d in demo:
        print("\nTEXT:", d["text"])
        for name in [k for k in d if k != "text"]:
            print(f"  {name:<10}", d[name])
    corpus_cmp = tc.compare_on_corpus(df["text"].tolist())
    print("\nCorpus-level comparison:")
    print(pd.DataFrame(corpus_cmp).to_string(index=False))
    challenges = tc.challenge_counts(df["text"].tolist())
    print("\nTokenization challenges in this data:")
    print(pd.DataFrame(challenges).to_string(index=False))
    regex_rows = regex_patterns.demo_counts(df["text"].tolist())
    print("\nRegular expressions (messages matched / real examples):")
    for r in regex_rows:
        print(f"  {r['name']:<22} {r['n_messages']:>5} msgs   e.g. {r['examples']}")
    R.update(tokenizer_demo=demo, tokenizer_corpus=corpus_cmp, tokenizer_challenges=challenges, regex=regex_rows)

    # ------------------------------------------------------------------ Task 4
    banner("TASK 4 - Cleaning & normalization pipeline")
    raw = df["text"].tolist()
    print("Pipeline order:", " -> ".join(cleaning.STEP_NAMES))
    steps_vocab = cleaning.vocab_by_step(raw)
    step_info = []
    just = {n: j for n, _, j in cleaning.STEPS}
    for row in steps_vocab:
        step_info.append({**row, "justification": just.get(row["step"], "Vocabulary of the untouched text.")})
        print(f"  {row['step']:<24} vocabulary = {row['vocab']}")
    v_before, v_after = steps_vocab[0]["vocab"], steps_vocab[-1]["vocab"]
    print(f"\nVOCABULARY BEFORE cleaning: {v_before}\nVOCABULARY AFTER  cleaning: {v_after}"
          f"\nReduction: {100 * (v_before - v_after) / v_before:.1f}%")
    explore.vocab_steps_plot(steps_vocab, config.FIGURES / "vocab_by_step.png")

    df["text_clean"] = [cleaning.clean_text(t) for t in raw]     # <- the ONE cleaning function
    print("\nBefore / after examples:")
    for i in df.sample(5, random_state=config.RANDOM_SEED).index:
        print("  RAW  :", df.at[i, "text"]); print("  CLEAN:", df.at[i, "text_clean"]); print()

    # stop-word check: do we delete words that separate spam from ham?
    from nltk.corpus import stopwords as _nltk_sw
    sw_full = set(_nltk_sw.words("english"))
    pre_sw = [cleaning.apply_steps(t, "strip_punctuation").split() for t in raw]
    spam_mask = (df["label"] == "spam").tolist()
    n_spam, n_ham = sum(spam_mask), len(spam_mask) - sum(spam_mask)
    dfc_s, dfc_h = Counter(), Counter()
    for toks, is_spam in zip(pre_sw, spam_mask):
        (dfc_s if is_spam else dfc_h).update(set(toks) & sw_full)
    sw_rows = []
    for w in sw_full:
        a, b = dfc_s[w], dfc_h[w]
        if a + b >= 20:
            sw_rows.append({"word": w, "spam_msgs": a, "ham_msgs": b,
                            "spam_rate": round(a / n_spam, 3), "ham_rate": round(b / n_ham, 3),
                            "ratio": round((a / n_spam + 1e-9) / (b / n_ham + 1e-9), 2),
                            "kept": w in cleaning.KEEP_WORDS or w not in cleaning.get_stopwords()})
    sw_rows.sort(key=lambda r: max(r["ratio"], 1 / max(r["ratio"], 1e-9)), reverse=True)
    sw_rows = sw_rows[:10]
    print("Stop words most different between spam and ham (kept = NOT removed by our pipeline):")
    print(pd.DataFrame(sw_rows).to_string(index=False))
    removed = sorted(sw_full - cleaning.get_stopwords())
    print(f"\nWords removed from NLTK stop list (kept in text): {removed}")

    # stemming vs lemmatization on the most frequent content words
    content = Counter(t for toks in (cleaning.apply_steps(x, "strip_punctuation").split() for x in raw)
                      for t in toks if t not in cleaning.get_stopwords() and t.isalpha() and len(t) > 2)
    top_words = [w for w, _ in content.most_common(600)]
    from nltk.corpus import wordnet
    stem_rows = [{"word": w, "stem": cleaning.stem_word(w), "lemma": cleaning.lemmatize_word(w)} for w in top_words]
    diff = [r for r in stem_rows if r["stem"] != r["lemma"]]
    non_word_stems = sum(1 for r in stem_rows if not wordnet.synsets(r["stem"]))
    non_word_lemmas = sum(1 for r in stem_rows if not wordnet.synsets(r["lemma"]))
    print("\nStemming (Porter) vs lemmatization (WordNet) - examples where they differ:")
    print(pd.DataFrame(diff[:20]).to_string(index=False))
    print(f"\nOf the {len(stem_rows)} most frequent words: stems not found in WordNet = {non_word_stems}, "
          f"lemmas not found in WordNet = {non_word_lemmas}")
    R.update(cleaning=dict(
        step_order=cleaning.STEP_NAMES, steps=step_info, vocab_before=v_before, vocab_after=v_after,
        reduction_pct=round(100 * (v_before - v_after) / v_before, 1),
        stopword_check=sw_rows, stopwords_kept=removed,
        stem_vs_lemma=diff[:12], n_compared=len(stem_rows),
        non_word_stems=non_word_stems, non_word_lemmas=non_word_lemmas,
        before_after=[{"raw": df.at[i, "text"], "clean": df.at[i, "text_clean"]}
                      for i in df.sample(4, random_state=7).index]))

    # ------------------------------------------------------------------ Task 5
    banner("TASK 5 - Save cleaned dataset (reproducible)")
    sig = pd.DataFrame([cleaning.extract_signals(t) for t in raw])
    out = pd.DataFrame({
        "label": df["label"], "label_num": df["label_num"],
        "text_raw": df["text"], "text_clean": df["text_clean"],
        "n_tokens_clean": df["text_clean"].str.split().str.len(),
        "caps_ratio": sig["caps_ratio"], "n_caps_words": sig["n_caps_words"], "n_exclaim": sig["n_exclaim"],
        "is_duplicate": df.duplicated(subset=["label", "text"]).astype(int),
    })
    out.to_csv(config.CLEAN_FILE, index=False, encoding="utf-8", lineterminator="\n")
    digest = sha256_of(config.CLEAN_FILE)
    n_empty = int((out["n_tokens_clean"] == 0).sum())
    print(f"Saved {len(out)} rows -> {config.CLEAN_FILE}")
    print(f"Columns: {list(out.columns)}")
    print(f"Rows whose cleaned text is empty: {n_empty}   Duplicate rows flagged: {int(out['is_duplicate'].sum())}")
    print(f"SHA-256 of output (must be identical on every machine): {digest}")
    R["output"] = {"path": str(config.CLEAN_FILE.relative_to(config.ROOT)), "n_rows": len(out),
                   "columns": list(out.columns), "n_empty_after_cleaning": n_empty,
                   "n_duplicates_flagged": int(out["is_duplicate"].sum()), "sha256": digest}

    config.RESULTS_JSON.write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults for the report saved to {config.RESULTS_JSON}")
    print("Phase 1 pipeline finished OK.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="path to a manually downloaded raw file")
    main(ap.parse_args().data)
