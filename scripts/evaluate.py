#!/usr/bin/env python3
"""Score streaming runs (RQ2-RQ4): transcript, timing, sign and cost metrics.

    python scripts/evaluate.py --runs runs --out results/streaming --usd-per-hour <your g4dn.xlarge price>

Writes
  per_meeting_transcript.csv  one row per (model, policy, mode, meeting)
  per_meeting_sign.csv        ... x mapper x schedule
  summary_transcript.csv / summary_sign.csv   mean and bootstrap 95% CI over meetings
  stats_*.csv                 Friedman + pairwise Wilcoxon (Holm) across policies
"""
import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2sbench.io import load_reference, load_run  # noqa: E402
from s2sbench.lexicon import Lexicon  # noqa: E402
from s2sbench.mapper import make_mapper  # noqa: E402
from s2sbench.metrics import map_reference, oracle_metrics, sign_metrics, transcript_metrics  # noqa: E402
from s2sbench.stats import bootstrap_ci, friedman, pairwise_wilcoxon  # noqa: E402

SCHEDULES = [("fifo", 1.0, None), ("fifo", 1.5, None), ("bounded", 1.0, 4.0), ("bounded", 1.5, 4.0),
             ("bounded", 1.5, 2.0), ("bounded", 1.5, 6.0)]   # L = 2 and 6 s: sensitivity of the bounded-lag results
ORACLE_ID = "O-oracle"  # perfect, zero-delay transcript of the same audio (reference row in the sign tables)
MAPPERS = ["M0", "M1", "M2"]


def sched_name(pol, rate, ml):
    return f"{pol}{'' if ml is None else f'{ml:g}'}@{rate:g}x"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--ref", default="data/ami/ref")
    ap.add_argument("--lexicon", default="data/lexicon.json")
    ap.add_argument("--lexicon-mode", default="active", choices=["active", "full"])
    ap.add_argument("--out", default="results/streaming")
    ap.add_argument("--usd-per-hour", type=float, default=float("nan"),
                    help="on-demand price of the GPU instance (state region and date in the paper)")
    ap.add_argument("--headline", nargs="*",
                    default=["P1-h1.0-c0.25", "P2-sil500", "P3-la-1.0", "P4-aa-1.0"])
    ap.add_argument("--no-oracle-rows", action="store_true",
                    help="do not add the perfect-transcript reference rows (policy O-oracle) to the sign tables")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lx = Lexicon.from_json(a.lexicon, mode=a.lexicon_mode)
    mappers = {m: make_mapper(m, lx) for m in MAPPERS}

    trows, srows = [], []
    ref_cache = {}
    oracle_done = set()
    files = sorted(glob.glob(os.path.join(a.runs, "*", "*", "*.json")))
    if not files:
        sys.exit(f"no run files under {a.runs}")
    for fp in files:
        run = load_run(fp)
        mt, dur = run["meeting"], run["duration"]
        key = (mt, run.get("start_seconds", 0.0), run.get("limit_seconds"))
        if key not in ref_cache:
            ref, segs = load_reference(os.path.join(a.ref, f"{mt}.json"), limit=run.get("limit_seconds"),
                                       start=run.get("start_seconds", 0.0) or 0.0)
            ref_items = {m: map_reference(mp, ref) for m, mp in mappers.items()}
            ref_cache[key] = (ref, segs, ref_items)
        ref, segs, ref_items = ref_cache[key]
        base = {"model": run["model"], "policy": run["policy_id"], "comp_aware": run["comp_aware"],
                "meeting": mt, "duration_s": dur}
        tm = transcript_metrics(ref, run["words"], dur, ref_segments=segs)
        rtf = run["compute_total"] / dur
        # recogniser invocations: simulator steps that did real work (a decode takes far longer
        # than 50 ms on any hardware we target; VAD-only steps take about 1 ms)
        rec_calls = sum(1 for it in run.get("iterations", []) if it.get("c", 0.0) > 0.05)
        tm.update({"rtf": rtf, "n_calls": run["n_calls"], "compute_s": run["compute_total"],
                   "rec_calls_per_min": rec_calls / (dur / 60.0),
                   "streams_per_gpu_ub": math.floor(1.0 / rtf) if rtf > 0 else float("nan"),
                   "usd_per_audio_hour_lb": rtf * a.usd_per_hour})
        trows.append({**base, **tm})
        for mname, mp in mappers.items():
            for pol, rate, ml in SCHEDULES:
                sm = sign_metrics(mp, ref, run["words"], dur, rate=rate, policy=pol, max_lag=ml,
                                  ref_items=ref_items[mname])
                srows.append({**base, "mapper": mname, "schedule": sched_name(pol, rate, ml), **sm})
        okey = (run["model"], run["comp_aware"]) + key
        if not a.no_oracle_rows and okey not in oracle_done:
            # the same excerpt with a perfect, zero-delay transcript ("O-oracle"), scored like a policy
            oracle_done.add(okey)
            obase = {**base, "policy": ORACLE_ID}
            for mname, mp in mappers.items():
                for pol, rate, ml in SCHEDULES:
                    om, played = oracle_metrics(mp, ref, dur, rate=rate, policy=pol, max_lag=ml)
                    n_play = sum(1 for p in played if p.item.played)
                    mins = dur / 60.0
                    srows.append({**obase, "mapper": mname, "schedule": sched_name(pol, rate, ml),
                                  "ser": 0.0, "spurious_fs_per_min": 0.0, "spurious_lex_per_min": 0.0,
                                  "rho_hyp": om["rho"], "rho_ref": om["rho"],
                                  "fs_items_per_min_hyp": om["fs_items"] / mins,
                                  "fs_items_per_min_ref": om["fs_items"] / mins,
                                  "letters_per_min_hyp": om["letters"] / mins,
                                  "played_items_per_min": om["delivered_items_per_min"],
                                  "delivered": om["delivered"],
                                  "delivered_correct_lex_per_min": om["delivered_lex_per_min"],
                                  "drop_rate": sum(1 for p in played if p.dropped) / max(n_play, 1),
                                  **{k: v for k, v in om.items() if k.startswith("sign_lag") or k == "overrun_s"}})
        print(f"{run['model']:>9s} {run['policy_id']:<16s} {'aware' if run['comp_aware'] else 'unaware'} {mt}: "
              f"WER {tm['wer']:.3f}  lag50 {tm['emit_lag_median']:.2f}s  RTF {rtf:.3f}")

    T = pd.DataFrame(trows)
    S = pd.DataFrame(srows)
    ex = [{"meeting": k[0], "start_s": k[1], "length_s": k[2], "ref_words": len(v[0]),
           "speakers": len({w.speaker for w in v[0]}), "ref_lex_M0": sum(1 for i in v[2]["M0"] if i.kind == "LEX"),
           "ref_fs_M0": sum(1 for i in v[2]["M0"] if i.kind == "FS")} for k, v in sorted(ref_cache.items())]
    pd.DataFrame(ex).to_csv(os.path.join(a.out, "excerpts.csv"), index=False)
    T.to_csv(os.path.join(a.out, "per_meeting_transcript.csv"), index=False)
    S.to_csv(os.path.join(a.out, "per_meeting_sign.csv"), index=False)

    def summarise(df, keys, cols):
        out = []
        for k, g in df.groupby(keys):
            row = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
            row["n_meetings"] = g["meeting"].nunique()
            for c in cols:
                if c in g:
                    m, lo, hi = bootstrap_ci(g[c].values)
                    row[c], row[c + "_lo"], row[c + "_hi"] = m, lo, hi
            out.append(row)
        return pd.DataFrame(out)

    tcols = ["wer", "tcorc_wer", "emit_lag_median", "emit_lag_p90", "emit_lag_le_2s", "frag_per_1k", "dup_per_1k",
             "nonspeech_tok_per_h", "rtf", "rec_calls_per_min", "usd_per_audio_hour_lb", "hyp_words", "wer_nref",
             "wer_sub", "wer_del", "wer_ins"]
    scols = ["ser", "spurious_fs_per_min", "spurious_lex_per_min", "rho_hyp", "rho_ref", "sign_lag_median",
             "sign_lag_p90", "sign_lag_le_4s", "delivered", "delivered_correct_lex_per_min", "drop_rate", "overrun_s",
             "fs_items_per_min_hyp", "fs_items_per_min_ref", "letters_per_min_hyp", "played_items_per_min"]
    summarise(T, ["model", "comp_aware", "policy"], tcols).to_csv(os.path.join(a.out, "summary_transcript.csv"),
                                                                 index=False)
    summarise(S, ["model", "comp_aware", "mapper", "schedule", "policy"], scols).to_csv(
        os.path.join(a.out, "summary_sign.csv"), index=False)

    # statistics across the headline policies (meetings = paired observations)
    st_rows = []
    for (model, ca), g in T.groupby(["model", "comp_aware"]):
        g = g[g["policy"].isin(a.headline)]
        piv = {c: g.pivot(index="meeting", columns="policy", values=c).dropna() for c in tcols if c in g}
        for c, pv in piv.items():
            if pv.shape[1] < 2 or pv.shape[0] < 5:
                continue
            tab = {p: pv[p].values for p in pv.columns}
            fr = friedman(tab)
            for r in pairwise_wilcoxon(tab):
                st_rows.append({"model": model, "comp_aware": ca, "metric": c, **fr, **r})
    for (model, ca, mname, sch), g in S.groupby(["model", "comp_aware", "mapper", "schedule"]):
        g = g[g["policy"].isin(a.headline)]
        for c in ["ser", "spurious_fs_per_min", "spurious_lex_per_min", "rho_hyp", "sign_lag_median", "sign_lag_p90",
                  "delivered", "delivered_correct_lex_per_min", "drop_rate"]:
            pv = g.pivot(index="meeting", columns="policy", values=c).dropna()
            if pv.shape[1] < 2 or pv.shape[0] < 5:
                continue
            tab = {p: pv[p].values for p in pv.columns}
            fr = friedman(tab)
            for r in pairwise_wilcoxon(tab):
                st_rows.append({"model": model, "comp_aware": ca, "mapper": mname, "schedule": sch,
                                "metric": c, **fr, **r})
    pd.DataFrame(st_rows).to_csv(os.path.join(a.out, "stats_pairwise.csv"), index=False)
    print(f"wrote results to {a.out}")


if __name__ == "__main__":
    main()
