#!/usr/bin/env python3
"""Run-to-run repeatability.

    python scripts/repeatability.py --a runs_cpu --b runs_cpu_repeat --out results/streaming/repeatability.csv

For every (model, policy, mode, meeting) present in both folders, the script compares the two runs:

* ``identical_output``: the committed words are the same (text, ASR start/end times and emission
  times, compared to 1 ms).  Computation-unaware runs should be identical: their output does not
  depend on how long each recognition call takes.  Computation-aware runs depend on measured
  compute times and may differ.
* differences in WER, median emission lag, number of committed words and real-time factor (RTF).
  The RTF is a timing measurement and varies even when the output is identical.
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2sbench.io import load_reference, load_run  # noqa: E402
from s2sbench.metrics import transcript_metrics  # noqa: E402


def _n_diff(a, b):
    import difflib

    sm = difflib.SequenceMatcher(a=_sig(a), b=_sig(b), autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")


def _sig(words):
    return [(w.text, round(w.start, 3), round(w.end, 3), None if w.emit is None else round(w.emit, 3))
            for w in words]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="runs_cpu")
    ap.add_argument("--b", default="runs_cpu_repeat")
    ap.add_argument("--ref", default="data/ami/ref")
    ap.add_argument("--out", default="results/streaming/repeatability.csv")
    a = ap.parse_args()
    rows = []
    for fb in sorted(glob.glob(os.path.join(a.b, "*", "*", "*.json"))):
        rel = os.path.relpath(fb, a.b)
        fa = os.path.join(a.a, rel)
        if not os.path.exists(fa):
            continue
        ra, rb = load_run(fa), load_run(fb)
        ref, segs = load_reference(os.path.join(a.ref, f"{ra['meeting']}.json"), limit=ra.get("limit_seconds"),
                                   start=ra.get("start_seconds", 0.0) or 0.0)
        ma = transcript_metrics(ref, ra["words"], ra["duration"])
        mb = transcript_metrics(ref, rb["words"], rb["duration"])
        rows.append({"model": ra["model"], "policy": ra["policy_id"], "comp_aware": ra["comp_aware"],
                     "meeting": ra["meeting"], "identical_output": _sig(ra["words"]) == _sig(rb["words"]),
                     "n_diff_words": _n_diff(ra["words"], rb["words"]),
                     "wer_a": ma["wer"], "wer_b": mb["wer"], "lag_a": ma["emit_lag_median"],
                     "lag_b": mb["emit_lag_median"], "rtf_a": ra["compute_total"] / ra["duration"],
                     "rtf_b": rb["compute_total"] / rb["duration"], "words_a": len(ra["words"]),
                     "words_b": len(rb["words"])})
    df = pd.DataFrame(rows)
    if df.empty:
        sys.exit("no overlapping runs")
    for m in ("wer", "lag", "rtf", "words"):
        df[f"d_{m}"] = df[f"{m}_b"] - df[f"{m}_a"]
    df["rtf_rel_diff"] = df["d_rtf"] / df["rtf_a"]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200):
        print(df[["model", "policy", "comp_aware", "meeting", "identical_output", "n_diff_words", "d_wer", "d_lag", "d_words",
                  "rtf_a", "rtf_b", "rtf_rel_diff"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
