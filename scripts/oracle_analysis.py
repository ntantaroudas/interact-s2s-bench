#!/usr/bin/env python3
"""RQ1 -- what the avatar is asked to sign when the transcript is perfect.

Uses only the AMI manual transcripts (no ASR, no GPU):
* lexical coverage of meeting speech by the 747-clip IS lexicon,
* most frequent uncovered words (vocabulary-expansion priorities),
* signing demand per second of speech (rho) for mappers M0-M2, clip timing
  modes (active/full) and playback rates,
* oracle sign lag and delivered fraction for FIFO and bounded-lag playback,
  with a zero-latency transcript and with a constant ASR delay.

    python scripts/oracle_analysis.py --ref data/ami/ref --lexicon data/lexicon.json --out results/oracle
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2sbench.ami import EVAL_MEETINGS  # noqa: E402
from s2sbench.io import load_reference  # noqa: E402
from s2sbench.lexicon import Lexicon  # noqa: E402
from s2sbench.mapper import make_mapper  # noqa: E402
from s2sbench.metrics import oracle_metrics  # noqa: E402
from s2sbench.normalize import strip_for_lookup  # noqa: E402
from s2sbench.stats import bootstrap_ci  # noqa: E402

MAPPERS = ["M0", "M1", "M2"]
MODES = ["active", "full"]
RATES = [1.0, 1.5, 2.0, 3.0]  # 3.0 = 30 fps clips stepped at 90 Hz
SCHEDULES = [("fifo", None), ("bounded", 2.0), ("bounded", 4.0), ("bounded", 6.0)]
ASR_DELAYS = [0.0, 2.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="data/ami/ref")
    ap.add_argument("--lexicon", default="data/lexicon.json")
    ap.add_argument("--durations", default="data/ami/eval_durations.json")
    ap.add_argument("--meetings", nargs="*", default=EVAL_MEETINGS)
    ap.add_argument("--out", default="results/oracle")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    durations = json.load(open(a.durations))["durations_s"] if os.path.exists(a.durations) else {}

    lex = {m: Lexicon.from_json(a.lexicon, mode=m) for m in MODES}
    rows, cov_rows = [], []
    oov_counter, oov_counter_m1 = collections.Counter(), collections.Counter()
    for mt in a.meetings:
        ref, _ = load_reference(os.path.join(a.ref, f"{mt}.json"))
        dur = durations.get(mt, max(w.end for w in ref))
        speech_min = dur / 60.0
        # ---- coverage (deployed rule: exact lower-cased word lookup) -------
        toks = [strip_for_lookup(w.text) for w in ref]
        toks = [t for t in toks if t]
        inlex = [t in lex["active"] for t in toks]
        for t, ok in zip(toks, inlex):
            if not ok:
                oov_counter[t] += 1
        m1 = make_mapper("M1", lex["active"])
        items_m1 = m1.map(ref, batch_by_emit=False)
        for it in items_m1:
            if it.kind == "FS":
                oov_counter_m1[it.label] += 1
        cov_rows.append({
            "meeting": mt, "duration_s": dur, "words": len(toks), "words_per_min": len(toks) / speech_min,
            "types": len(set(toks)), "token_coverage_M0": float(np.mean(inlex)),
            "type_coverage_M0": float(np.mean([t in lex["active"] for t in set(toks)])),
        })
        # ---- demand, lag, delivered ----------------------------------------
        for mode in MODES:
            for mp_name in MAPPERS:
                mp = make_mapper(mp_name, lex[mode])
                for rate in RATES:
                    for pol, ml in SCHEDULES:
                        for d in ASR_DELAYS:
                            if mode == "full" and (pol != "fifo" or d != 0.0):
                                continue  # 'full' timing only for the demand figure
                            res, _ = oracle_metrics(mp, ref, dur, rate=rate, policy=pol, max_lag=ml, asr_delay=d)
                            res.update({"meeting": mt, "mode": mode, "mapper": mp_name, "rate": rate,
                                        "schedule": pol if ml is None else f"{pol}{ml:g}", "asr_delay": d,
                                        "duration_s": dur, "items_per_min": res["n_items"] / speech_min,
                                        "lex_per_min": res["lex_items"] / speech_min,
                                        "fs_per_min": res["fs_items"] / speech_min,
                                        "letters_per_min": res["letters"] / speech_min})
                            rows.append(res)
        print(f"{mt}: {len(toks)} words, token coverage {np.mean(inlex):.3f}")

    df = pd.DataFrame(rows)
    cov = pd.DataFrame(cov_rows)
    df.to_csv(os.path.join(a.out, "oracle_per_meeting.csv"), index=False)
    cov.to_csv(os.path.join(a.out, "coverage_per_meeting.csv"), index=False)
    pd.DataFrame(oov_counter.most_common(200), columns=["word", "count"]).to_csv(
        os.path.join(a.out, "oov_top200_M0.csv"), index=False)
    pd.DataFrame(oov_counter_m1.most_common(200), columns=["word", "count"]).to_csv(
        os.path.join(a.out, "fingerspelled_top200_M1.csv"), index=False)

    # ---- summaries: mean and bootstrap 95% CI across meetings ---------------
    def summarise(g, cols):
        out = {}
        for c in cols:
            m, lo, hi = bootstrap_ci(g[c].values)
            out[c] = m
            out[c + "_lo"] = lo
            out[c + "_hi"] = hi
        return pd.Series(out)

    cols = ["rho", "items_per_min", "lex_per_min", "fs_per_min", "letters_per_min", "sign_lag_median",
            "sign_lag_p90", "sign_lag_le_4s", "delivered", "delivered_lex_per_min", "overrun_s"]
    summ = df.groupby(["mode", "mapper", "rate", "schedule", "asr_delay"]).apply(
        lambda g: summarise(g, cols), include_groups=False).reset_index()
    summ.to_csv(os.path.join(a.out, "oracle_summary.csv"), index=False)
    cs = {c: bootstrap_ci(cov[c].values) for c in ["words_per_min", "token_coverage_M0", "type_coverage_M0"]}
    # share of the fingerspelling load carried by the N most frequent missing words
    def topn_share(counter, ns=(25, 50, 100, 200)):
        tot = sum(counter.values())
        mc = [c for _, c in counter.most_common()]
        return {str(n): (sum(mc[:n]) / tot if tot else float("nan")) for n in ns}, tot

    fs_share_m1, fs_total_m1 = topn_share(oov_counter_m1)
    oov_share_m0, oov_total_m0 = topn_share(oov_counter)
    # dataset statistics: all words incl. fillers, and the share of the kept (non-filler) words that
    # overlap with another speaker (overlap flags from the AMI loader)
    import json as _json
    ds = {"words_all": 0, "fillers": 0, "overlap_words": 0}
    from s2sbench.normalize import FILLERS
    for mt in a.meetings:
        raw = _json.load(open(os.path.join(a.ref, f"{mt}.json")))["words"]
        kept = [w for w in raw if strip_for_lookup(w["text"]) not in FILLERS]
        ds["words_all"] += len(raw)
        ds["fillers"] += len(raw) - len(kept)
        ds["overlap_words"] += sum(1 for w in kept if (w.get("meta") or {}).get("overlap"))
    ds["overlap_share"] = ds["overlap_words"] / (ds["words_all"] - ds["fillers"])
    ds["filler_share"] = ds["fillers"] / ds["words_all"]
    lx = Lexicon.from_json(a.lexicon)
    from s2sbench.lexicon import LETTERS
    timing = {}
    for name, sel in (("signs", lambda e: e.key not in LETTERS and not e.key.isdigit()),
                      ("letters", lambda e: e.key in LETTERS)):
        act = np.array([e.active_frames for e in lx.by_key.values() if sel(e)]) / lx.fps
        full = np.array([e.frames for e in lx.by_key.values() if sel(e)]) / lx.fps
        timing[name] = {"n": int(len(act)),
                        "active_s_median": float(np.median(act)), "active_s_q1": float(np.percentile(act, 25)),
                        "active_s_q3": float(np.percentile(act, 75)), "full_s_median": float(np.median(full)),
                        "full_s_q1": float(np.percentile(full, 25)), "full_s_q3": float(np.percentile(full, 75))}
    total = {
        "dataset": ds,
        "fs_topN_share_M1": fs_share_m1, "fs_total_M1": fs_total_m1,
        "oov_topN_share_M0": oov_share_m0, "oov_total_M0": oov_total_m0,
        "clip_timing_at_30fps": timing,
        "meetings": len(cov), "hours": float(cov["duration_s"].sum() / 3600), "words": int(cov["words"].sum()),
        "overall_token_coverage_M0": float((cov["token_coverage_M0"] * cov["words"]).sum() / cov["words"].sum()),
        "per_meeting_ci": {k: {"mean": v[0], "lo": v[1], "hi": v[2]} for k, v in cs.items()},
        "lexicon": Lexicon.from_json(a.lexicon).stats(),
    }
    json.dump(total, open(os.path.join(a.out, "coverage_summary.json"), "w"), indent=1)
    print(json.dumps(total, indent=1))


if __name__ == "__main__":
    main()
