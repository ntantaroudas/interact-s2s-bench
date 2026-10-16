#!/usr/bin/env python3
"""Build the manuscript Markdown from the template and the results files.

    python manuscript/build_paper.py            # writes manuscript/INTERACT_StreamingS2S_ORE_v3.md

* ``{{name}}``        -> a number computed from results/ (RQ1 from results/oracle, RQ2-4 from results/streaming)
* ``{{TABLE:name}}``  -> a Markdown table built from results/
* ``[@key]``, ``[@a; @b]`` -> numbered citations [n] in order of first appearance; the reference list is
  generated from references.yaml.
Anything that cannot be computed yet is rendered as **[PENDING: ...]** so it is easy to find.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ORACLE = os.path.join(ROOT, "results", "oracle")
STREAM = os.path.join(ROOT, "results", "streaming")
HEADLINE = ["P1-h1.0-c0.25", "P2-sil500", "P3-la-1.0", "P4-aa-1.0"]
POLICY_LABEL = {
    "O-oracle": "O perfect transcript",
    "P1v-h1.0-c0.25-vad": "P1v = P1 + VAD gate",
    "P1-h1.0-c0.25": "P1 fixed 1 s (deployed)", "P1-h1.0-c0": "P1 fixed 1 s, no context",
    "P1-h2.0-c0.25": "P1 fixed 2 s + 0.25 s", "P1-h3.0-c0.25": "P1 fixed 3 s + 0.25 s",
    "P2-sil500": "P2 pause/VAD", "P2-sil300": "P2 pause/VAD, 300 ms",
    "P3-la-1.0": "P3 LocalAgreement-2", "P3-la-2.0": "P3 LocalAgreement-2, 2 s",
    "P3-la-1.0-fd": "P3 LocalAgreement-2, 1 s + final decode",
    "P4-aa-1.0": "P4 AlignAtt", "P4-aa-2.0": "P4 AlignAtt, 2 s",
}


def pending(name):
    return f"**[PENDING: {name}]**"


def f(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:,.{nd}f}"


def ci(row, col, nd=2, unit=""):
    return f"{f(row[col], nd)}{unit} (95% CI {f(row[col + '_lo'], nd)}–{f(row[col + '_hi'], nd)}{unit})"


SHY = "­"   # soft hyphen: lets Word break "LocalAgreement-2" as "Local-/Agreement-2" in a narrow cell


def md_table(df: pd.DataFrame) -> str:
    # headers say "per min" rather than "/min" so that narrow columns can wrap between words;
    # the column widths are set later for all tables by layout_tables()
    cols = list(df.columns)
    heads = [str(c).replace("/min", " per min") for c in cols]
    out = ["| " + " | ".join(heads) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(out)


# ----------------------------------------------------------- table layout --
TABLE_CHARS = 100   # characters per line of table text (9 pt) across the text width


def _cells(line):
    s = line.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") else s
    return [c.strip() for c in s.split("|")]


def _visible(s):
    s = re.sub(r"\*+", "", s)
    return s.replace(SHY, "")


def _longest_piece(s):
    """Longest piece of a cell that Word cannot break: words, split again after hyphens and soft hyphens
    (a piece of one or two characters stays with the piece before it, e.g. "Agreement-2")."""
    best = 0
    for tok in re.split(r"\s+", re.sub(r"\*+", "", s)):
        pieces = []
        for p in re.split(r"(?<=-)|­", tok):
            if pieces and len(p) <= 2:
                pieces[-1] += p
            elif p:
                pieces.append(p)
        best = max([best] + [len(p) for p in pieces])
    return best


def _widths(rows):
    """Relative column widths (dash counts) for Word/PDF output, like an automatic table layout: every
    column is at least as wide as its longest unbreakable piece plus the cell padding, and the remaining
    width goes to the columns with long text."""
    n = len(rows[0])
    wmin = [max(6, max(_longest_piece(r[j]) for r in rows) + 3) for j in range(n)]
    wnat = [max(wmin[j], min(60, max(len(_visible(r[j])) for r in rows) + 3)) for j in range(n)]
    total = max(TABLE_CHARS, sum(wmin))
    extra = total - sum(wmin)
    want = [b - a for a, b in zip(wmin, wnat)]
    if sum(want) and extra <= sum(want):
        w = [a + extra * d / sum(want) for a, d in zip(wmin, want)]
    else:
        rest = extra - sum(want)
        w = [b + rest * b / sum(wnat) for b in wnat]
    return [max(a, int(round(x))) for a, x in zip(wmin, w)]


def layout_tables(md: str) -> str:
    lines = md.split("\n")
    out, i = [], 0
    while i < len(lines):
        if (lines[i].lstrip().startswith("|") and i + 1 < len(lines)
                and re.fullmatch(r"\s*\|?(\s*:?-+:?\s*\|)+\s*:?-*:?\s*\|?\s*", lines[i + 1])):
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            rows = [_cells(lines[i])] + [_cells(x) for x in lines[i + 2:j]]
            rows = [[c.replace("LocalAgreement", f"Local{SHY}Agreement") for c in r] for r in rows]
            w = _widths(rows)
            out.append("| " + " | ".join(rows[0]) + " |")
            out.append("|" + "|".join("-" * x for x in w) + "|")
            out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)


# ------------------------------------------------------------------- RQ1 --
def oracle_vars():
    V, T = {}, {}
    cs_path = os.path.join(ORACLE, "coverage_summary.json")
    if not os.path.exists(cs_path):
        return V, T
    cs = json.load(open(cs_path))
    s = pd.read_csv(os.path.join(ORACLE, "oracle_summary.csv"))
    cov = pd.read_csv(os.path.join(ORACLE, "coverage_per_meeting.csv"))
    ds, lx, tm = cs["dataset"], cs["lexicon"], cs["clip_timing_at_30fps"]
    V.update({
        "ds_meetings": str(cs["meetings"]), "ds_hours": f(cs["hours"], 2), "ds_words_all": f"{ds['words_all']:,}",
        "ds_words": f"{cs['words']:,}", "ds_fillers": f"{ds['fillers']:,}", "ds_filler_pct": f(100 * ds["filler_share"], 1),
        "ds_overlap_pct": f(100 * ds["overlap_share"], 1),
        "ds_wpm": f(cs["per_meeting_ci"]["words_per_min"]["mean"], 0),
        "ds_wpm_ci": f"{f(cs['per_meeting_ci']['words_per_min']['lo'], 0)}–{f(cs['per_meeting_ci']['words_per_min']['hi'], 0)}",
        "lex_files": str(lx["files"]), "lex_keys": str(lx["unique_keys"]), "lex_collisions": str(len(lx["collisions"])),
        "lex_letters": str(lx["letters"]), "lex_numbers": str(lx["numbers"]), "lex_multi": str(lx["multiword"]),
        "lex_unreach": str(len(lx["unreachable_single_token"])),
        "lex_signs": str(lx["unique_keys"] - lx["letters"] - lx["numbers"]),
        "t_sign_act_med": f(tm["signs"]["active_s_median"], 2),
        "t_sign_act_iqr": f"{f(tm['signs']['active_s_q1'], 2)}–{f(tm['signs']['active_s_q3'], 2)}",
        "t_sign_full_med": f(tm["signs"]["full_s_median"], 2),
        "t_sign_full_iqr": f"{f(tm['signs']['full_s_q1'], 2)}–{f(tm['signs']['full_s_q3'], 2)}",
        "t_let_act_med": f(tm["letters"]["active_s_median"], 2),
        "t_let_act_iqr": f"{f(tm['letters']['active_s_q1'], 2)}–{f(tm['letters']['active_s_q3'], 2)}",
        "t_let_full_med": f(tm["letters"]["full_s_median"], 2),
        "cov_tok_m0": f(100 * cs["overall_token_coverage_M0"], 1),
        "cov_type_m0": f(100 * cs["per_meeting_ci"]["type_coverage_M0"]["mean"], 1),
        "oov_top100_m0": f(100 * cs["oov_topN_share_M0"]["100"], 1), "oov_top200_m0": f(100 * cs["oov_topN_share_M0"]["200"], 1),
        "fs_top100_m1": f(100 * cs["fs_topN_share_M1"]["100"], 1), "fs_top200_m1": f(100 * cs["fs_topN_share_M1"]["200"], 1),
    })

    def row(mode, mapper, rate, sched="fifo", delay=0.0):
        r = s[(s["mode"] == mode) & (s["mapper"] == mapper) & (s["rate"] == rate) & (s["schedule"] == sched)
              & (s["asr_delay"] == delay)]
        return r.iloc[0]

    for m in ("M0", "M1", "M2"):
        r = row("active", m, 1.0)
        V[f"{m.lower()}_items_pm"] = f(r["items_per_min"], 1)
        V[f"{m.lower()}_lex_pm"] = f(r["lex_per_min"], 1)
        V[f"{m.lower()}_fs_pm"] = f(r["fs_per_min"], 1)
        V[f"{m.lower()}_letters_pm"] = f(r["letters_per_min"], 0)
        V[f"{m.lower()}_fs_share"] = f(100 * r["fs_per_min"] / r["items_per_min"], 1) if r["items_per_min"] else "0"
        for rate in (1.0, 1.5, 2.0, 3.0):
            rr = row("active", m, rate)
            key = f"{m}_r{str(rate).replace('.', '')}"
            V[f"rho_{key}"] = ci(rr, "rho", 2)
            V[f"rhoci_{key}"] = f"95% CI {f(rr['rho_lo'], 2)}–{f(rr['rho_hi'], 2)}"
            V[f"rhoval_{key}"] = f(rr["rho"], 2)
            V[f"lag_{key}"] = f(rr["sign_lag_median"], 1)
            V[f"lagp90_{key}"] = f(rr["sign_lag_p90"], 1)
            V[f"lagmin_{key}"] = f(rr["sign_lag_median"] / 60.0, 1)
            V[f"laghr_{key}"] = f(rr["sign_lag_median"] / 3600.0, 2)
            for d in (0.0, 2.0):
                b = row("active", m, rate, "bounded4", d)
                dk = f"{key}_d{int(d)}"
                V[f"del4lex_{dk}"] = ci(b, "delivered_lex_per_min", 1)
                V[f"del4lexval_{dk}"] = f(b["delivered_lex_per_min"], 1)
                V[f"del4pct_{dk}"] = f(100 * b["delivered_lex_per_min"] / r["lex_per_min"], 0) if r["lex_per_min"] else "–"
                V[f"del4frac_{dk}"] = f(100 * b["delivered"], 0)
        rf = row("full", m, 1.0)
        V[f"rho_full_{m}_r10"] = ci(rf, "rho", 1)

    # per-meeting spread behind the mean for M2 at 1.5x (FIFO)
    pm_path = os.path.join(ORACLE, "oracle_per_meeting.csv")
    if os.path.exists(pm_path):
        pm = pd.read_csv(pm_path)
        d = pm[(pm["mode"] == "active") & (pm["mapper"] == "M2") & (pm["rate"] == 1.5) & (pm["schedule"] == "fifo")
               & (pm["asr_delay"] == 0)]
        en, non = d[d["meeting"].str.startswith("EN")], d[~d["meeting"].str.startswith("EN")]
        V["en_rho_lo"], V["en_rho_hi"] = f(en["rho"].min(), 2), f(en["rho"].max(), 2)
        V["en_lag_lo"], V["en_lag_hi"] = f(en["sign_lag_median"].min(), 0), f(en["sign_lag_median"].max(), 0)
        V["nonen_lag"], V["nonen_rho_hi"] = f(non["sign_lag_median"].mean(), 1), f(non["rho"].max(), 2)

    # Table: oracle summary (active timing)
    rows = []
    for m in ("M0", "M1", "M2"):
        for rate in (1.0, 1.5, 2.0, 3.0):
            r = row("active", m, rate)
            b = row("active", m, rate, "bounded4", 0.0)
            rows.append({"Mapper": m, "Rate": f"{rate:g}×", "Signed items/min": f(r["lex_per_min"] + r["fs_per_min"], 1),
                         "FS items/min": f(r["fs_per_min"], 1), "Letters/min": f(r["letters_per_min"], 0),
                         "ρ (95% CI)": f"{f(r['rho'])} ({f(r['rho_lo'])}–{f(r['rho_hi'])})",
                         "FIFO median lag (s)": f(r["sign_lag_median"], 1), "FIFO p90 lag (s)": f(r["sign_lag_p90"], 1),
                         "Lex. signs/min delivered, ≤4 s": f(b["delivered_lex_per_min"], 1)})
    T["oracle"] = md_table(pd.DataFrame(rows))

    # Table: dataset by meeting series
    cov["series"] = cov["meeting"].str[:2]
    g = cov.groupby("series").agg(meetings=("meeting", "count"), hours=("duration_s", lambda x: x.sum() / 3600),
                                   words=("words", "sum"), wpm=("words_per_min", "mean"),
                                   cov=("token_coverage_M0", "mean")).reset_index()
    names = {"ES": "ES (scenario)", "IS": "IS (scenario)", "TS": "TS (scenario)", "EN": "EN (non-scenario)"}
    rows = [{"Series": names.get(r["series"], r["series"]), "Meetings": int(r["meetings"]), "Hours": f(r["hours"], 2),
             "Words": f"{int(r['words']):,}", "Words/min": f(r["wpm"], 0), "M0 token coverage (%)": f(100 * r["cov"], 1)}
            for _, r in g.iterrows()]
    rows.append({"Series": "**All**", "Meetings": int(cov.shape[0]), "Hours": f(cov["duration_s"].sum() / 3600, 2),
                 "Words": f"{int(cov['words'].sum()):,}", "Words/min": f(cov["words_per_min"].mean(), 0),
                 "M0 token coverage (%)": f(100 * cs["overall_token_coverage_M0"], 1)})
    T["dataset"] = md_table(pd.DataFrame(rows))

    # Table: lexicon
    T["lexicon"] = md_table(pd.DataFrame([
        {"Property": "Landmark files (one per video clip)", "Value": lx["files"]},
        {"Property": "Distinct lookup keys (case-insensitive)", "Value": f"{lx['unique_keys']} ({len(lx['collisions'])} case collisions: "
                                                                     + ", ".join(lx["collisions"]) + ")"},
        {"Property": "Lexical signs (excluding letters and numbers)", "Value": V["lex_signs"]},
        {"Property": "Manual-alphabet letters / number signs", "Value": f"{lx['letters']} / {lx['numbers']} (1–10)"},
        {"Property": "Multi-word entries", "Value": lx["multiword"]},
        {"Property": "Entries unreachable by single-word lookup", "Value": ", ".join(lx["unreachable_single_token"])},
        {"Property": "Sign clip, hand-visible span: median (IQR)", "Value": f"{V['t_sign_act_med']} s ({V['t_sign_act_iqr']} s)"},
        {"Property": "Sign clip, full length: median (IQR)", "Value": f"{V['t_sign_full_med']} s ({V['t_sign_full_iqr']} s)"},
        {"Property": "Letter clip, hand-visible span: median (IQR)", "Value": f"{V['t_let_act_med']} s ({V['t_let_act_iqr']} s)"},
        {"Property": "Letter clip, full length: median", "Value": f"{V['t_let_full_med']} s"},
    ]))

    # Table: most frequent uncovered words
    a = pd.read_csv(os.path.join(ORACLE, "oov_top200_M0.csv")).head(20)
    b = pd.read_csv(os.path.join(ORACLE, "fingerspelled_top200_M1.csv")).head(20)
    rows = [{"Rank": i + 1, "M0: word not in lexicon (count)": f"{a.iloc[i]['word']} ({a.iloc[i]['count']:,})",
             "M1: word still fingerspelled (count)": f"{b.iloc[i]['word']} ({b.iloc[i]['count']:,})"} for i in range(20)]
    T["oov"] = md_table(pd.DataFrame(rows))
    return V, T


# --------------------------------------------------------------- RQ2-RQ4 --
SHORT = {"P1-h1.0-c0.25": "P1", "P1v-h1.0-c0.25-vad": "P1v", "P2-sil500": "P2", "P3-la-1.0": "P3",
         "P4-aa-1.0": "P4", "O-oracle": "O"}
ENGINE = {"P1": "faster-whisper, int8", "P1v": "faster-whisper, int8", "P2": "faster-whisper, int8",
          "P3": "faster-whisper, int8", "P4": "PyTorch, float32"}
T_METRICS = [  # (variable stem, column, scale, decimals)
    ("wer", "wer", 100, 1), ("tcorc", "tcorc_wer", 100, 1), ("lag50", "emit_lag_median", 1, 2),
    ("lag90", "emit_lag_p90", 1, 2), ("lagle2", "emit_lag_le_2s", 100, 0), ("frag", "frag_per_1k", 1, 1),
    ("dup", "dup_per_1k", 1, 1), ("nonsp", "nonspeech_tok_per_h", 1, 0), ("rtf", "rtf", 1, 2),
    ("calls", "rec_calls_per_min", 1, 1), ("hypw", "hyp_words", 1, 0), ("ins", "wer_ins", 1, 1),
    ("del", "wer_del", 1, 1), ("sub", "wer_sub", 1, 1), ("nref", "wer_nref", 1, 0),
]
S_SETS = [  # (mapper, schedule, tag)
    ("M0", "fifo@1x", "m0"), ("M1", "fifo@1x", "m1"), ("M2", "fifo@1x", "m2f"), ("M2", "fifo@1.5x", "m2f15"),
    ("M0", "bounded4@1x", "m0b"), ("M1", "bounded4@1.5x", "m1b15"), ("M2", "bounded4@1x", "m2b1"),
    ("M2", "bounded4@1.5x", "m2"), ("M0", "bounded4@1.5x", "m0b15"),
    ("M2", "bounded2@1.5x", "m2l2"), ("M2", "bounded6@1.5x", "m2l6"),
]
S_METRICS = [
    ("ser", "ser", 100, 1), ("sfs", "spurious_fs_per_min", 1, 1), ("slex", "spurious_lex_per_min", 1, 1),
    ("rho", "rho_hyp", 1, 2), ("rhoref", "rho_ref", 1, 2), ("fsm", "fs_items_per_min_hyp", 1, 1),
    ("let", "letters_per_min_hyp", 1, 0), ("slag", "sign_lag_median", 1, 1), ("slag90", "sign_lag_p90", 1, 1),
    ("dlex", "delivered_correct_lex_per_min", 1, 1), ("dshare", "delivered", 100, 0), ("drop", "drop_rate", 100, 0),
    ("play", "played_items_per_min", 1, 1), ("over", "overrun_s", 1, 0),
]
STAT_METRICS = [("wer", "WER"), ("emit_lag_median", "Emission lag, median"), ("frag_per_1k", "FRAG"),
                ("nonspeech_tok_per_h", "Non-speech tokens/h"), ("spurious_fs_per_min", "Spurious FS items/min"),
                ("spurious_lex_per_min", "Spurious lexical signs/min"), ("rho_hyp", "Signing demand ρ"),
                ("delivered_correct_lex_per_min", "Correct lexical signs/min delivered"),
                ("sign_lag_median", "Sign lag, median")]


def _fmt(v, scale, nd):
    return f(v * scale if v is not None and np.isfinite(v) else np.nan, nd)


def stream_tables():
    V, T = {}, {}
    tp = os.path.join(STREAM, "summary_transcript.csv")
    if not os.path.exists(tp):
        return V, T
    t = pd.read_csv(tp)
    s = pd.read_csv(os.path.join(STREAM, "summary_sign.csv"))
    stp = os.path.join(STREAM, "stats_pairwise.csv")
    try:
        st = pd.read_csv(stp) if os.path.exists(stp) else None
    except pd.errors.EmptyDataError:  # fewer than 5 meetings: no tests yet
        st = None
    # the paper's streaming experiment: one model, one simulation mode
    model = t["model"].mode().iloc[0]
    aware = bool(t["comp_aware"].mode().iloc[0])
    t = t[(t["model"] == model) & (t["comp_aware"] == aware)]
    s = s[(s["model"] == model) & (s["comp_aware"] == aware)]
    V["st_model"] = f"Whisper {model}"
    V["st_n"] = str(int(t["n_meetings"].max()))

    def order(df):
        df = df.copy()
        df["_o"] = df["policy"].map({p: i for i, p in enumerate(SHORT)}).fillna(99)
        return df.sort_values("_o").drop(columns="_o")

    def label(pid):
        return POLICY_LABEL.get(pid, pid)

    for _, r in t.iterrows():
        k = SHORT.get(r["policy"], r["policy"])
        if r.get("wer_nref", 0) > 0:
            for err in ("ins", "del", "sub"):
                V[f"{err}pct_{k}"] = f(100 * r[f"wer_{err}"] / r["wer_nref"], 1)
        for stem, col, sc, nd in T_METRICS:
            if col in r:
                V[f"{stem}_{k}"] = _fmt(r[col], sc, nd)
                if col + "_lo" in r:
                    V[f"{stem}ci_{k}"] = f"95% CI {_fmt(r[col + '_lo'], sc, nd)}–{_fmt(r[col + '_hi'], sc, nd)}"
    for mapper, sched, tag in S_SETS:
        d = s[(s["mapper"] == mapper) & (s["schedule"] == sched)]
        for _, r in d.iterrows():
            k = SHORT.get(r["policy"], r["policy"])
            for stem, col, sc, nd in S_METRICS:
                if col in r:
                    V[f"{stem}_{tag}_{k}"] = _fmt(r[col], sc, nd)
                    if col + "_lo" in r:
                        V[f"{stem}ci_{tag}_{k}"] = f"95% CI {_fmt(r[col + '_lo'], sc, nd)}–{_fmt(r[col + '_hi'], sc, nd)}"
        o = d[d["policy"] == "O-oracle"]
        if not o.empty:
            o_dlex = float(o.iloc[0]["delivered_correct_lex_per_min"])
            for _, r in d.iterrows():
                k = SHORT.get(r["policy"], r["policy"])
                if o_dlex > 0:
                    V[f"dlexpct_{tag}_{k}"] = f(100 * r["delivered_correct_lex_per_min"] / o_dlex, 0)
                if r["rho_ref"] > 0:
                    V[f"rhox_{tag}_{k}"] = f(r["rho_hyp"] / r["rho_ref"], 2)

    # ---- tables
    rows = []
    for _, r in order(t).iterrows():
        rows.append({"Policy": label(r["policy"]),
                     "WER (%)": f"{_fmt(r['wer'], 100, 1)} ({_fmt(r['wer_lo'], 100, 1)}–{_fmt(r['wer_hi'], 100, 1)})",
                     "tcORC-WER (%)": _fmt(r.get("tcorc_wer", np.nan), 100, 1),
                     "Emission lag, median (s)": f"{f(r['emit_lag_median'], 2)} ({f(r['emit_lag_median_lo'], 2)}–{f(r['emit_lag_median_hi'], 2)})",
                     "Emission lag, p90 (s)": f(r["emit_lag_p90"], 2),
                     "FRAG /1k": f(r["frag_per_1k"], 1), "DUP /1k": f(r["dup_per_1k"], 1),
                     "Non-speech tokens/h": f(r["nonspeech_tok_per_h"], 0)})
    T["rq2"] = md_table(pd.DataFrame(rows))
    T["cost"] = md_table(pd.DataFrame([{
        "Policy": label(r["policy"]), "Inference engine": ENGINE.get(SHORT.get(r["policy"], ""), ""),
        "Recogniser calls per audio minute": f(r["rec_calls_per_min"], 1),
        "RTF on 2 vCPUs (95% CI)": f"{f(r['rtf'], 2)} ({f(r['rtf_lo'], 2)}–{f(r['rtf_hi'], 2)})"}
        for _, r in order(t).iterrows()]))

    def sign_table(mapper, schedule, cols):
        d = order(s[(s["mapper"] == mapper) & (s["schedule"] == schedule)])
        rows = []
        for _, r in d.iterrows():
            row = {"Policy": label(r["policy"])}
            for c, lab, sc, nd, with_ci in cols:
                v = _fmt(r[c], sc, nd)
                if with_ci and c + "_lo" in r and r["policy"] != "O-oracle":
                    v += f" ({_fmt(r[c + '_lo'], sc, nd)}–{_fmt(r[c + '_hi'], sc, nd)})"
                row[lab] = v
            rows.append(row)
        return md_table(pd.DataFrame(rows)) if rows else pending(f"{mapper} {schedule}")

    T["rq3_m0"] = sign_table("M0", "fifo@1x", [
        ("ser", "SER (%)", 100, 1, True), ("spurious_fs_per_min", "Spurious FS items/min", 1, 1, True),
        ("spurious_lex_per_min", "Spurious lexical signs/min", 1, 1, False),
        ("fs_items_per_min_hyp", "FS items/min", 1, 1, False), ("letters_per_min_hyp", "Letters/min", 1, 0, False),
        ("rho_hyp", "ρ at 1×", 1, 2, False)])
    T["rq3_m1"] = sign_table("M1", "fifo@1x", [
        ("ser", "SER (%)", 100, 1, True), ("spurious_fs_per_min", "Spurious FS items/min", 1, 1, True),
        ("spurious_lex_per_min", "Spurious lexical signs/min", 1, 1, False),
        ("fs_items_per_min_hyp", "FS items/min", 1, 1, False), ("rho_hyp", "ρ at 1×", 1, 2, False)])
    T["rq4"] = sign_table("M2", "bounded4@1.5x", [
        ("delivered_correct_lex_per_min", "Correct lexical signs/min", 1, 1, True),
        ("delivered", "Reference signs delivered (%)", 100, 0, False),
        ("sign_lag_median", "Sign lag, median (s)", 1, 2, False), ("sign_lag_p90", "Sign lag, p90 (s)", 1, 2, False),
        ("drop_rate", "Items dropped (%)", 100, 0, False), ("spurious_lex_per_min", "Spurious lexical signs/min", 1, 1, False)])
    T["rq4_m0"] = sign_table("M0", "fifo@1x", [
        ("rho_hyp", "ρ", 1, 2, False), ("sign_lag_median", "Sign lag, median (s)", 1, 1, True),
        ("sign_lag_p90", "Sign lag, p90 (s)", 1, 1, False),
        ("delivered_correct_lex_per_min", "Correct lexical signs/min", 1, 1, False),
        ("overrun_s", "Signing after the excerpt ends (s)", 1, 0, False)])

    # ---- statistics
    if st is not None and not st.empty:
        st = st[(st["model"] == model) & (st["comp_aware"] == aware)]
        rows = []
        for col, lab in STAT_METRICS:
            sel = st[st["metric"] == col]
            if col in ("spurious_fs_per_min", "spurious_lex_per_min", "rho_hyp"):
                sel, ctx = sel[(sel["mapper"] == "M0") & (sel["schedule"] == "fifo@1x")], "M0, FIFO 1×"
            elif col in ("delivered_correct_lex_per_min", "sign_lag_median"):
                sel, ctx = sel[(sel["mapper"] == "M2") & (sel["schedule"] == "bounded4@1.5x")], "M2, bounded 4 s, 1.5×"
            else:
                ctx = "transcript"
            if sel.empty:
                continue
            fr = sel.iloc[0]
            pairs = []
            for _, r in sel.iterrows():
                a, b = SHORT.get(r["a"], r["a"]), SHORT.get(r["b"], r["b"])
                V[f"p_{col}_{a}{b}"] = f"{r['p_holm']:.2g}"
                V[f"md_{col}_{a}{b}"] = f(r["median_diff"], 3)
                star = "**" if r["p_holm"] < 0.05 else ""
                pairs.append(f"{star}{a}–{b}: {r['p_holm']:.2g}{star}")
            V[f"frp_{col}"] = f"{fr['friedman_p']:.2g}"
            V[f"frchi_{col}"] = f(fr["friedman_chi2"], 1)
            rows.append({"Metric": lab, "Setting": ctx,
                         "Friedman χ²(3), p": f"{f(fr['friedman_chi2'], 1)}, {fr['friedman_p']:.2g}",
                         "Pairwise Wilcoxon, Holm-adjusted p": "; ".join(pairs)})
        T["stats"] = md_table(pd.DataFrame(rows)) if rows else pending("stats")

    # ---- do transcript metrics predict sign metrics?  (Spearman over meeting x policy)
    ptp, psp = os.path.join(STREAM, "per_meeting_transcript.csv"), os.path.join(STREAM, "per_meeting_sign.csv")
    if os.path.exists(ptp) and os.path.exists(psp):
        from scipy.stats import spearmanr

        pt, ps = pd.read_csv(ptp), pd.read_csv(psp)
        pt = pt[(pt["model"] == model) & (pt["comp_aware"] == aware) & pt["policy"].isin(HEADLINE)]
        ps = ps[(ps["model"] == model) & (ps["comp_aware"] == aware) & ps["policy"].isin(HEADLINE)]
        m0 = ps[(ps["mapper"] == "M0") & (ps["schedule"] == "fifo@1x")][
            ["policy", "meeting", "ser", "spurious_fs_per_min", "spurious_lex_per_min", "rho_hyp"]]
        m2 = ps[(ps["mapper"] == "M2") & (ps["schedule"] == "bounded4@1.5x")][
            ["policy", "meeting", "delivered_correct_lex_per_min", "sign_lag_median"]]
        d = pt.merge(m0, on=["policy", "meeting"]).merge(m2, on=["policy", "meeting"])
        V["rs_n"] = str(len(d))
        # within-meeting association: centre every metric on its meeting mean (policy-driven variation only)
        cols = ["wer", "emit_lag_median", "emit_lag_p90", "nonspeech_tok_per_h", "frag_per_1k", "ser",
                "spurious_fs_per_min", "spurious_lex_per_min", "delivered_correct_lex_per_min", "sign_lag_median"]
        dc = d.copy()
        dc[cols] = d[cols] - d.groupby("meeting")[cols].transform("mean")
        for name, x, y in [("dlex_wer", "delivered_correct_lex_per_min", "wer"),
                           ("dlex_lag", "delivered_correct_lex_per_min", "emit_lag_median"),
                           ("dlex_lag90", "delivered_correct_lex_per_min", "emit_lag_p90"),
                           ("sfs_nonsp", "spurious_fs_per_min", "nonspeech_tok_per_h"),
                           ("sfs_frag", "spurious_fs_per_min", "frag_per_1k"),
                           ("ser_wer", "ser", "wer")]:
            if len(dc) >= 5:
                r, pv = spearmanr(dc[x], dc[y])
                V[f"rsw_{name}"], V[f"rswp_{name}"] = f(r, 2), f"{pv:.2g}"
        # the excerpt where P1 did worst, and P1v there
        p1 = pt[pt["policy"] == "P1-h1.0-c0.25"]
        if not p1.empty:
            w = p1.loc[p1["wer"].idxmax()]
            V["p1_worst_meeting"], V["p1_worst_wer"] = w["meeting"], f(100 * w["wer"], 0)
            V["p1_worst_hypw"] = str(int(w["hyp_words"]))
            V["p1_worst_nonsp"] = f(w["nonspeech_tok_per_h"], 0)
            ptall = pd.read_csv(ptp)
            v = ptall[(ptall["policy"] == "P1v-h1.0-c0.25-vad") & (ptall["meeting"] == w["meeting"])
                      & (ptall["model"] == model) & (ptall["comp_aware"] == aware)]
            if not v.empty:
                V["p1v_worst_wer"] = f(100 * v.iloc[0]["wer"], 0)
            others = pt[(pt["meeting"] == w["meeting"]) & (pt["policy"] != "P1-h1.0-c0.25")]["wer"]
            if len(others):
                V["others_worst_wer_lo"], V["others_worst_wer_hi"] = f(100 * others.min(), 0), f(100 * others.max(), 0)
        # spurious fingerspelling of P1 relative to the mean of P2-P4
        sfs = m0.groupby("policy")["spurious_fs_per_min"].mean()
        rest = [sfs.get(x) for x in ("P2-sil500", "P3-la-1.0", "P4-aa-1.0") if x in sfs]
        if "P1-h1.0-c0.25" in sfs and rest:
            V["sfs_ratio_P1"] = f(sfs["P1-h1.0-c0.25"] / np.mean(rest), 1)
        for name, x, y in [("sfs_nonsp", "spurious_fs_per_min", "nonspeech_tok_per_h"),
                           ("sfs_frag", "spurious_fs_per_min", "frag_per_1k"),
                           ("sfs_wer", "spurious_fs_per_min", "wer"),
                           ("slex_nonsp", "spurious_lex_per_min", "nonspeech_tok_per_h"),
                           ("ser_wer", "ser", "wer"),
                           ("dlex_wer", "delivered_correct_lex_per_min", "wer"),
                           ("dlex_lag", "delivered_correct_lex_per_min", "emit_lag_median"),
                           ("slag_lag", "sign_lag_median", "emit_lag_median")]:
            if len(d) >= 5:
                r, pv = spearmanr(d[x], d[y])
                V[f"rs_{name}"], V[f"rsp_{name}"] = f(r, 2), f"{pv:.2g}"

    # ---- the derived VAD-gated P1 (P1v) against P1: paired Wilcoxon over meetings
    if os.path.exists(ptp) and os.path.exists(psp):
        from scipy.stats import wilcoxon

        pt, ps = pd.read_csv(ptp), pd.read_csv(psp)
        pt = pt[(pt["model"] == model) & (pt["comp_aware"] == aware)]
        ps = ps[(ps["model"] == model) & (ps["comp_aware"] == aware)]
        pairs = [("wer", pt, None), ("nonspeech_tok_per_h", pt, None), ("frag_per_1k", pt, None),
                 ("spurious_fs_per_min", ps, ("M0", "fifo@1x")), ("spurious_lex_per_min", ps, ("M0", "fifo@1x")),
                 ("delivered_correct_lex_per_min", ps, ("M2", "bounded4@1.5x"))]
        for col, df, sel in pairs:
            d = df if sel is None else df[(df["mapper"] == sel[0]) & (df["schedule"] == sel[1])]
            pv = d.pivot_table(index="meeting", columns="policy", values=col)
            if {"P1-h1.0-c0.25", "P1v-h1.0-c0.25-vad"} <= set(pv.columns):
                x, y = pv["P1-h1.0-c0.25"].values, pv["P1v-h1.0-c0.25-vad"].values
                if len(x) >= 5 and not np.allclose(x, y):
                    V[f"p1v_p_{col}"] = f"{wilcoxon(x, y).pvalue:.2g}"
                if np.nanmean(x) > 0:
                    V[f"p1v_red_{col}"] = f(100 * (1 - np.nanmean(y) / np.nanmean(x)), 0)
        if "P1v-h1.0-c0.25-vad" in set(pt["policy"]):
            a1 = pt[pt["policy"] == "P1-h1.0-c0.25"]["rec_calls_per_min"].mean()
            a2 = pt[pt["policy"] == "P1v-h1.0-c0.25-vad"]["rec_calls_per_min"].mean()
            V["p1v_windows_kept"] = f(100 * a2 / a1, 0) if a1 else "–"

    # ---- which items make up the spurious signing (scripts/spurious_breakdown.py)
    sip, scp = os.path.join(STREAM, "spurious_items_M0.csv"), os.path.join(STREAM, "spurious_causes_M0.csv")
    if os.path.exists(sip):
        si = pd.read_csv(sip)
        for pid, g in si.groupby("policy"):
            k = SHORT.get(pid, pid)
            tot = g["count"].sum()
            ty = g[g["label"].str.lower().isin(["thank", "you", "thanks"])]["count"].sum()
            V[f"thankyou_share_{k}"] = f(100 * ty / tot, 0) if tot else "0"
            V[f"thankyou_n_{k}"] = str(int(ty))
            V[f"spur_n_{k}"] = str(int(tot))
            top = g.sort_values("count", ascending=False).head(5)
            V[f"spur_top_{k}"] = ", ".join(f"{('fingerspelled ' if r['kind'] == 'FS' else '')}{str(r['label']).upper()} ({int(r['count'])})"
                                           for _, r in top.iterrows())
    if os.path.exists(scp):
        sc = pd.read_csv(scp)
        for pid, g in sc.groupby("policy"):
            k = SHORT.get(pid, pid)
            tot = g["per_min"].sum()
            for cause in ("non-speech", "fragment", "other"):
                v = g[g["cause"] == cause]["per_min"].sum()
                V[f"cause_{cause.replace('-', '')}_{k}"] = f(v, 1)
                V[f"causepct_{cause.replace('-', '')}_{k}"] = f(100 * v / tot, 0) if tot else "0"

    if st is not None and not st.empty:
        for sch, tag in (("bounded2@1.5x", "l2"), ("bounded6@1.5x", "l6"), ("bounded4@1x", "b1"), ("bounded4@1.5x", "l4")):
            r = st[(st["metric"] == "delivered_correct_lex_per_min") & (st["mapper"] == "M2") & (st["schedule"] == sch)
                   & (st["a"] == "P1-h1.0-c0.25") & (st["b"] == "P4-aa-1.0")]
            if not r.empty:
                V[f"p_dlex_{tag}_P1P4"] = f"{r.iloc[0]['p_holm']:.2g}"
    if os.path.exists(scp):
        sc = pd.read_csv(scp)
        for pid, g in sc[sc["kind"] == "FS"].groupby("policy"):
            k = SHORT.get(pid, pid)
            for cause in ("non-speech", "fragment", "other"):
                V[f"causefs_{cause.replace('-', '')}_{k}"] = f(g[g["cause"] == cause]["per_min"].sum(), 1)
    if os.path.exists(psp):
        ps_all = pd.read_csv(psp)
        m0p = ps_all[(ps_all["model"] == model) & (ps_all["comp_aware"] == aware) & (ps_all["mapper"] == "M0")
                     & (ps_all["schedule"] == "fifo@1x")].pivot_table(index="meeting", columns="policy", values="rho_hyp")
        if {"P1-h1.0-c0.25", "O-oracle"} <= set(m0p.columns):
            V["p1_rho_above_O_n"] = str(int((m0p["P1-h1.0-c0.25"] > m0p["O-oracle"]).sum()))
    if os.path.exists(ptp) and "p1_worst_meeting" in V:
        pt_all = pd.read_csv(ptp)
        w = pt_all[(pt_all["model"] == model) & (pt_all["comp_aware"] == aware)].pivot_table(
            index="meeting", columns="policy", values="wer")
        if {"P1-h1.0-c0.25", "P1v-h1.0-c0.25-vad"} <= set(w.columns):
            dd = 100 * (w["P1-h1.0-c0.25"] - w["P1v-h1.0-c0.25-vad"])
            V["p1v_wer_red_pp"] = f(dd.mean(), 1)
            V["p1v_wer_red_excl_pp"] = f(dd.drop(V["p1_worst_meeting"]).mean(), 1)

    # ---- aggregates quoted in the text (min/max over policies, ratios)
    def num(key):
        try:
            return float(str(V[key]).replace(",", ""))
        except (KeyError, ValueError):
            return None

    def agg(out_key, keys, fn, nd):
        vals = [num(k) for k in keys]
        vals = [v for v in vals if v is not None]
        if vals:
            V[out_key] = f(fn(vals), nd)

    P234 = ("P2", "P3", "P4")
    P1234 = ("P1", "P2", "P3", "P4")
    agg("subpct_max_P2P3P4", [f"subpct_{k}" for k in P234], max, 1)
    agg("inspct_max_P2P3P4", [f"inspct_{k}" for k in P234], max, 1)
    agg("nonsp_max_P2P3P4", [f"nonsp_{k}" for k in P234], max, 0)
    agg("frag_min", [f"frag_{k}" for k in P1234], min, 1)
    agg("frag_max", [f"frag_{k}" for k in P1234], max, 1)
    for tag in ("m0", "m1"):
        for stem, nd in (("ser", 1), ("sfs", 1), ("rho", 2), ("slex", 1)):
            agg(f"{stem}_{tag}_min_P2P3P4", [f"{stem}_{tag}_{k}" for k in P234], min, nd)
            agg(f"{stem}_{tag}_max_P2P3P4", [f"{stem}_{tag}_{k}" for k in P234], max, nd)
    agg("slag_m0_min_P", [f"slag_m0_{k}" for k in P1234], min, 1)
    agg("slag_m0_max_P", [f"slag_m0_{k}" for k in P1234], max, 1)
    agg("thankyou_share_max_P2P3P4", [f"thankyou_share_{k}" for k in P234], max, 0)
    rt = [num(f"rtf_{k}") for k in P1234]
    if all(v for v in rt):
        V["rtf_ratio"] = f(max(rt) / min(rt), 1)
    for k in P1234 + ("O",):
        if num(f"slag_m0_{k}") is not None:
            V[f"slag0_m0_{k}"] = f(num(f"slag_m0_{k}"), 0)
            V[f"slagmin_m0_{k}"] = f(num(f"slag_m0_{k}") / 60.0, 1)
    agg("slag0_m0_min_P", [f"slag_m0_{k}" for k in P1234], min, 0)
    agg("slag0_m0_max_P", [f"slag_m0_{k}" for k in P1234], max, 0)
    if num("slex_m2_P1") and num("slex_m2_P4") is not None:
        V["slex_m2_pct_P4P1"] = f(100 * num("slex_m2_P4") / num("slex_m2_P1"), 0)
    # is P2 last at every lag limit?
    lastP2 = []
    for tag in ("m2l2", "m2", "m2l6"):
        vals = {k: num(f"dlex_{tag}_{k}") for k in P1234}
        if all(v is not None for v in vals.values()):
            lastP2.append(min(vals, key=vals.get) == "P2")
    if lastP2:
        V["p2_last_all_L"] = "yes" if all(lastP2) else "no"
    if num("dlex_m2_P4") and num("dlex_m2_P2"):
        V["dlex_ratio_P4P2"] = f(num("dlex_m2_P4") / num("dlex_m2_P2"), 1)
    # is the ranking of the policies by correct signs delivered the same for L = 2, 4 and 6 s?
    ranks = []
    for tag in ("m2l2", "m2", "m2l6"):
        vals = {k: num(f"dlex_{tag}_{k}") for k in P1234}
        if all(v is not None for v in vals.values()):
            ranks.append(tuple(sorted(vals, key=lambda k: -vals[k])))
    if len(ranks) == 3:
        V["rank_L_same"] = "the same" if len(set(ranks)) == 1 else "not the same"
        V["rank_L_order"] = " > ".join(ranks[1])
    pw = [num(f"p_wer_{a}") for a in ("P2P3", "P2P4", "P3P4")]
    if all(v is not None for v in pw):
        V["p_wer_min_P2P3P4"] = f"{min(pw):.2g}"
    pl = [num(f"p_emit_lag_median_{a}") for a in ("P1P2", "P1P3", "P1P4", "P2P3", "P2P4", "P3P4")]
    if all(v is not None for v in pl):
        V["p_lag_max"] = f"{max(pl):.2g}"

    # ---- excerpts and repeatability
    exp = os.path.join(STREAM, "excerpts.csv")
    if os.path.exists(exp):
        ex = pd.read_csv(exp)
        if "p1_worst_meeting" in V:
            m = ex[ex["meeting"] == V["p1_worst_meeting"]]
            if not m.empty:
                V["p1_worst_refw"] = str(int(m.iloc[0]["ref_words"]))
        mins = ex["length_s"].sum() / 60.0
        V.update({"ex_n": str(len(ex)), "ex_len_s": f(ex["length_s"].median(), 0), "ex_minutes": f(mins, 0),
                  "ex_words": f"{int(ex['ref_words'].sum()):,}", "ex_words_min": str(int(ex["ref_words"].min())),
                  "ex_words_max": str(int(ex["ref_words"].max())), "ex_wpm": f(ex["ref_words"].sum() / mins, 0)})
        T["excerpts"] = md_table(pd.DataFrame([{"Meeting": r["meeting"], "Excerpt start (s)": f(r["start_s"], 2),
                                                "Reference words": int(r["ref_words"]), "Speakers": int(r["speakers"])}
                                               for _, r in ex.iterrows()]))
    rp = os.path.join(STREAM, "repeatability.csv")
    if os.path.exists(rp):
        r = pd.read_csv(rp)
        n, same = len(r), int(r["identical_output"].sum())
        V["rep_n"], V["rep_same"] = str(n), str(same)
        V["rep_rtf_diff"] = f(100 * r["rtf_rel_diff"].abs().mean(), 1)
        V["rep_rtf_maxdiff"] = f(100 * r["rtf_rel_diff"].abs().max(), 0)
        for _, row in r.iterrows():
            k = SHORT.get(row["policy"], row["policy"])
            V[f"rep_same_{k}"] = "yes" if row["identical_output"] else "no"
            if "n_diff_words" in row:
                V[f"rep_ndiff_{k}"] = str(int(row["n_diff_words"]))
            V[f"rep_nwords_{k}"] = str(int(row["words_a"]))
            V[f"rep_dwer_{k}"] = f(100 * abs(row["d_wer"]), 1)
        if same == n:
            V["repeat_sentence"] = (f"In all {n} repeated runs the committed words, their timestamps and their emission "
                                    f"times were identical to the first run; only the measured compute time varied "
                                    f"(mean absolute difference {V['rep_rtf_diff']}% of the real-time factor, maximum "
                                    f"{V['rep_rtf_maxdiff']}%).")
        else:
            V["repeat_sentence"] = (f"{same} of the {n} repeated runs reproduced the committed words, timestamps and "
                                    f"emission times exactly; the differences in the other runs are reported in the "
                                    f"extended data.")
    return V, T


def load_blocks(path):
    """Hand-written text blocks: <!--BLOCK:NAME--> ... (until the next block marker)."""
    if not os.path.exists(path):
        return {}
    txt = open(path, encoding="utf-8").read()
    parts = re.split(r"<!--BLOCK:([A-Z_]+)-->", txt)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}


# ---------------------------------------------------------------- render --
def render(tmpl: str, V: dict, T: dict, refs: dict) -> str:
    def tab(m):
        return T.get(m.group(1), pending(f"table {m.group(1)} — run the streaming experiment and scripts/evaluate.py"))

    txt = re.sub(r"\{\{TABLE:([A-Za-z0-9_]+)\}\}", tab, tmpl)
    txt = re.sub(r"\{\{([A-Za-z0-9_]+)\}\}", lambda m: V.get(m.group(1), pending(m.group(1))), txt)
    order = []

    def cite(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        nums = []
        for k in keys:
            if k not in refs:
                raise KeyError(f"unknown reference key: {k}")
            if k not in order:
                order.append(k)
            nums.append(order.index(k) + 1)
        nums = sorted(set(nums))
        parts, i = [], 0
        while i < len(nums):  # compress ranges 3,4,5 -> 3–5
            j = i
            while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
                j += 1
            parts.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ", ".join(str(n) for n in nums[i:j + 1]))
            i = j + 1
        return "[" + ", ".join(parts) + "]"

    txt = re.sub(r"(ρs? = |= )-(\d)", "\\1\u2212\\2", txt)   # typographic minus for negative coefficients
    body, sep, tail = txt.partition("<!--REFERENCES-->")
    body = re.sub(r"\[(@[^\]]+)\]", cite, body)
    tail = re.sub(r"\[(@[^\]]+)\]", cite, tail)
    reflist = "\n".join(f"{i + 1}. {refs[k]}" for i, k in enumerate(order))
    return layout_tables(body) + reflist + tail


def main():
    tmpl = open(os.path.join(HERE, "paper_template.md"), encoding="utf-8").read()
    blocks = load_blocks(os.path.join(HERE, "results_text.md"))
    for _ in range(3):  # blocks may contain other blocks
        tmpl = re.sub(r"\{\{([A-Z_]+)\}\}", lambda m: blocks.get(m.group(1), m.group(0)), tmpl)
    refs = {k: " ".join(v.split()) for k, v in yaml.safe_load(open(os.path.join(HERE, "references.yaml"))).items()}
    V, T = oracle_vars()
    V2, T2 = stream_tables()
    V.update(V2)
    T.update(T2)
    json.dump(V, open(os.path.join(ROOT, "results", "paper_variables.json"), "w"), indent=1, ensure_ascii=False)
    out = render(tmpl, V, T, refs)
    dst = os.path.join(HERE, "INTERACT_StreamingS2S_ORE_v3.md")
    open(dst, "w", encoding="utf-8").write(out)
    n_pending = len(re.findall(r"\[PENDING:", out))
    print(f"wrote {dst}  ({len(out.split()):,} words, {n_pending} pending items)")
    unused = sorted(set(refs) - set(re.findall(r"@([A-Za-z0-9_]+)", tmpl)))
    if unused:
        print("references not cited:", ", ".join(unused))


if __name__ == "__main__":
    sys.exit(main())
