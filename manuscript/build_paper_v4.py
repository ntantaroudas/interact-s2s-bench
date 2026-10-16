#!/usr/bin/env python3
"""Build version 4 of the manuscript (plain-language version, numbers in tables).

    python manuscript/build_paper_v4.py      # writes manuscript/INTERACT_StreamingS2S_ORE_v5.md

Uses the same results files as build_paper.py and its variables ({{name}}), table layout and citation
numbering ([@key] -> [n] in order of first appearance), with the plain-language tables of version 4 and the
semicolon-free reference list references_v4.yaml.  The output must not contain any semicolon.
"""
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_paper as bp  # noqa: E402

ORACLE, STREAM = bp.ORACLE, bp.STREAM
f, md_table = bp.f, bp.md_table

LABEL = {"P1-h1.0-c0.25": "P1 fixed 1 s (deployed)", "P1v-h1.0-c0.25-vad": "P1v = P1 + VAD gate",
         "P2-sil500": "P2 pause-based", "P3-la-1.0": "P3 LocalAgreement-2", "P4-aa-1.0": "P4 AlignAtt",
         "O-oracle": "O perfect transcript"}
ORDER = list(LABEL)
SHORT = bp.SHORT
ENGINE = {"P1": "faster-whisper, 8-bit integer", "P1v": "faster-whisper, 8-bit integer",
          "P2": "faster-whisper, 8-bit integer", "P3": "faster-whisper, 8-bit integer", "P4": "PyTorch, 32-bit float"}


def order(df):
    df = df[df["policy"].isin(ORDER)].copy()
    df["_o"] = df["policy"].map({p: i for i, p in enumerate(ORDER)})
    return df.sort_values("_o").drop(columns="_o")


def withci(r, c, sc=1.0, nd=1):
    return f"{f(r[c] * sc, nd)} ({f(r[c + '_lo'] * sc, nd)}–{f(r[c + '_hi'] * sc, nd)})"


def kv(rows):
    return md_table(pd.DataFrame([{"Item": a, "Value": b} for a, b in rows]))


def tables(V):
    T = {}
    s_or = pd.read_csv(os.path.join(ORACLE, "oracle_summary.csv"))
    cov = pd.read_csv(os.path.join(ORACLE, "coverage_per_meeting.csv"))

    def orow(mode, mapper, rate, sched="fifo", delay=0.0):
        r = s_or[(s_or["mode"] == mode) & (s_or["mapper"] == mapper) & (s_or["rate"] == rate)
                 & (s_or["schedule"] == sched) & (s_or["asr_delay"] == delay)]
        return r.iloc[0]

    # ---- Table 1: the meetings
    cov["series"] = cov["meeting"].str[:2]
    g = cov.groupby("series").agg(meetings=("meeting", "count"), hours=("duration_s", lambda x: x.sum() / 3600),
                                   words=("words", "sum"), wpm=("words_per_min", "mean"),
                                   cov=("token_coverage_M0", "mean")).reset_index()
    names = {"EN": "EN (real meetings)", "ES": "ES (role-played)", "IS": "IS (role-played)", "TS": "TS (role-played)"}
    rows = [{"Series": names.get(r["series"], r["series"]), "Meetings": int(r["meetings"]), "Hours": f(r["hours"], 2),
             "Words": f"{int(r['words']):,}", "Words per minute": f(r["wpm"], 0),
             "Words with a dictionary sign (%)": f(100 * r["cov"], 1)} for _, r in g.iterrows()]
    rows.append({"Series": "**All**", "Meetings": int(cov.shape[0]), "Hours": V["ds_hours"], "Words": V["ds_words"],
                 "Words per minute": f(cov["words_per_min"].mean(), 0), "Words with a dictionary sign (%)": V["cov_tok_m0"]})
    T["dataset4"] = md_table(pd.DataFrame(rows))

    # ---- Table 2: data preparation
    T["dataprep"] = kv([
        ("Audio", "headset mix (Mix-Headset), the 16 kHz sum of the speakers' headset microphones"),
        ("Perfect transcripts", "AMI manual word alignment, release 1.6.2, with a start and end time for every word"),
        ("Hesitations and back-channels removed (for example uh, um, mm-hmm)",
         f"{V['ds_fillers']} of {V['ds_words_all']} word tokens ({V['ds_filler_pct']}%)"),
        ("Words spoken while another speaker was talking", f"{V['ds_overlap_pct']}% of the {V['ds_words']} remaining words"),
        ("Word order when speakers overlap", "whole AMI speaker segments ordered by start time, words by time within a segment"),
        ("Streaming excerpts", f"one {V['ex_len_s']} s excerpt from the middle of each meeting, starting at (meeting length − {V['ex_len_s']} s)/2, "
                               "which avoids the openings and closings. Words kept if their midpoint falls inside, times shifted to the "
                               "start of the excerpt, speaker segments rebuilt from the kept words"),
        ("Excerpt material", f"{V['ex_minutes']} min, {V['ex_words']} reference words, {V['ex_words_min']} to "
                             f"{V['ex_words_max']} per excerpt, about {V['ex_wpm']} words per minute"),
    ])

    # ---- Table 3: the dictionary
    cs = __import__("json").load(open(os.path.join(ORACLE, "coverage_summary.json")))
    lx = cs["lexicon"]
    T["lexicon4"] = md_table(pd.DataFrame([
        {"Property": "Content of each file", "Value": "MediaPipe Holistic landmarks for every frame: 33 body points and 2 × 21 hand points"},
        {"Property": "Landmark files (one per dictionary video)", "Value": lx["files"]},
        {"Property": "Lookup entries (upper and lower case treated alike)",
         "Value": f"{lx['unique_keys']} ({len(lx['collisions'])} pairs of files share an entry: " + ", ".join(lx["collisions"]) + ")"},
        {"Property": "Dictionary signs (letters and numbers excluded)", "Value": V["lex_signs"]},
        {"Property": "Letters of the manual alphabet / number signs", "Value": f"{lx['letters']} / {lx['numbers']} (1 to 10)"},
        {"Property": "Multi-word entries", "Value": lx["multiword"]},
        {"Property": "Entries a single-word lookup can never reach", "Value": ", ".join(lx["unreachable_single_token"])},
        {"Property": "Sign clip, part with visible hands: median (IQR)", "Value": f"{V['t_sign_act_med']} s ({V['t_sign_act_iqr']} s)"},
        {"Property": "Sign clip, full length: median (IQR)", "Value": f"{V['t_sign_full_med']} s ({V['t_sign_full_iqr']} s)"},
        {"Property": "Letter clip, part with visible hands: median (IQR)", "Value": f"{V['t_let_act_med']} s ({V['t_let_act_iqr']} s)"},
        {"Property": "Letter clip, full length: median", "Value": f"{V['t_let_full_med']} s"},
    ]))

    # ---- Table 6: simulation, computer, repeatability
    same = [k for k in ("P2", "P3", "P4") if V.get(f"rep_same_{k}") == "yes"]
    T["simsetup"] = kv([
        ("Audio input", "real-time stream in 40 ms packets, the method works on all audio received so far"),
        ("Recogniser", "Whisper small (244 million parameters), English"),
        ("Timing", "delays caused by the method itself, computing time not included (computation-unaware mode)"),
        ("Warm-up", "one recogniser call before each excerpt, not scored"),
        ("Computer for computing cost", "cloud virtual machine with 2 virtual processors (Intel Xeon, 2.1 GHz, AVX-512) "
                                        "and 7 GB of memory, one job at a time with 2 threads"),
        ("Avatar playback speeds", "1×, 1.5×, 2× and 3× the 30 frames per second of the clips"),
        ("Lag limits", "2, 4 and 6 s, measured from the end of the word as timed by the recogniser"),
        ("Repeatability check (excerpt ES2004a, all four methods run again)",
         f"{', '.join(same[:-1])} and {same[-1]} identical. P1: {V['rep_ndiff_P1']} of {V['rep_nwords_P1']} words differed, "
         f"WER changed by {V['rep_dwer_P1']} percentage points. Computing time varied by up to {V['rep_rtf_maxdiff']}%"),
    ])

    # ---- Table 8: what a perfect transcript asks for
    r1 = {m: orow("active", m, 1.0) for m in ("M0", "M1", "M2")}
    rf = {m: orow("full", m, 1.0) for m in ("M0", "M1", "M2")}
    cols = {"M0": "M0 deployed", "M1": "M1 reduced", "M2": "M2 no fingerspelling"}

    def line(label, fn):
        d = {"Measure": label}
        d.update({cols[m]: fn(m) for m in cols})
        return d

    T["rq1load"] = md_table(pd.DataFrame([
        line("Sign items per minute", lambda m: f(r1[m]["lex_per_min"] + r1[m]["fs_per_min"], 1)),
        line("of which dictionary signs", lambda m: f(r1[m]["lex_per_min"], 1)),
        line("of which fingerspelled words", lambda m: f(r1[m]["fs_per_min"], 1)),
        line("Fingerspelled share of the sign items (%)",
             lambda m: f(100 * r1[m]["fs_per_min"] / (r1[m]["lex_per_min"] + r1[m]["fs_per_min"]), 1)),
        line("Letters to fingerspell per minute", lambda m: f(r1[m]["letters_per_min"], 0)),
        line("Signing demand at 1× (95% CI)", lambda m: withci(r1[m], "rho", 1, 2)),
        line("Signing demand at 1×, full clips (95% CI)", lambda m: withci(rf[m], "rho", 1, 1)),
    ]))

    # ---- Table 9: outcomes per mapping rule and playback speed
    rows = []
    for m in ("M0", "M1", "M2"):
        for rate in (1.0, 1.5, 2.0, 3.0):
            r = orow("active", m, rate)
            b0, b2 = orow("active", m, rate, "bounded4", 0.0), orow("active", m, rate, "bounded4", 2.0)
            avail = r1[m]["lex_per_min"]
            rows.append({"Rule": m, "Speed": f"{rate:g}×", "Signing demand (95% CI)": withci(r, "rho", 1, 2),
                         "All in order: median sign delay (s)": f(r["sign_lag_median"], 1),
                         "All in order: 90th percentile (s)": f(r["sign_lag_p90"], 1),
                         "Lag limit 4 s: dictionary signs delivered per min (95% CI)": withci(b0, "delivered_lex_per_min", 1, 1),
                         "Lag limit 4 s: share of available signs (%)": f(100 * b0["delivered_lex_per_min"] / avail, 0),
                         "Lag limit 4 s, recognition delay 2 s: signs per min": f(b2["delivered_lex_per_min"], 1)})
    T["rq1rates"] = md_table(pd.DataFrame(rows))

    # ---- Table 10: missing words
    a = pd.read_csv(os.path.join(ORACLE, "oov_top200_M0.csv")).head(20)
    b = pd.read_csv(os.path.join(ORACLE, "fingerspelled_top200_M1.csv")).head(20)
    rows = [{"Rank": str(i + 1), "M0: word without a dictionary entry (count)": f"{a.iloc[i]['word']} ({a.iloc[i]['count']:,})",
             "M1: word still fingerspelled (count)": f"{b.iloc[i]['word']} ({b.iloc[i]['count']:,})"} for i in range(20)]
    rows.append({"Rank": "Top 100", "M0: word without a dictionary entry (count)": f"{V['oov_top100_m0']}% of all fingerspelled words",
                 "M1: word still fingerspelled (count)": f"{V['fs_top100_m1']}% of the remaining fingerspelled words"})
    rows.append({"Rank": "Top 200", "M0: word without a dictionary entry (count)": f"{V['oov_top200_m0']}% of all fingerspelled words",
                 "M1: word still fingerspelled (count)": f"{V['fs_top200_m1']}% of the remaining fingerspelled words"})
    T["oov4"] = md_table(pd.DataFrame(rows))

    # ---- streaming tables
    t = pd.read_csv(os.path.join(STREAM, "summary_transcript.csv"))
    s = pd.read_csv(os.path.join(STREAM, "summary_sign.csv"))
    model = t["model"].mode().iloc[0]
    aware = bool(t["comp_aware"].mode().iloc[0])
    t = t[(t["model"] == model) & (t["comp_aware"] == aware)]
    s = s[(s["model"] == model) & (s["comp_aware"] == aware)]

    pt = pd.read_csv(os.path.join(STREAM, "per_meeting_transcript.csv"))
    pt = pt[(pt["model"] == model) & (pt["comp_aware"] == aware)]
    err = {pid: {k: float((100 * g[f"wer_{k}"] / g["wer_nref"]).mean()) for k in ("sub", "del", "ins")}
           for pid, g in pt.groupby("policy")}
    rows = []
    for _, r in order(t).iterrows():
        e = err[r["policy"]]
        rows.append({"Method": LABEL[r["policy"]], "WER (%)": withci(r, "wer", 100, 1),
                     "Overlap-tolerant WER (%)": f(100 * r["tcorc_wer"], 1),
                     "Substituted per 100 words": f(e["sub"], 1),
                     "Deleted per 100 words": f(e["del"], 1),
                     "Inserted per 100 words": f(e["ins"], 1),
                     "Repeated words per 1,000": f(r["dup_per_1k"], 1),
                     "Fragments per 1,000": f(r["frag_per_1k"], 1),
                     "Words where nobody spoke, per hour": f(r["nonspeech_tok_per_h"], 0)})
    T["rq2acc"] = md_table(pd.DataFrame(rows))

    rows = []
    for _, r in order(t).iterrows():
        k = SHORT[r["policy"]]
        rows.append({"Method": LABEL[r["policy"]], "Median word delay, s (95% CI)": withci(r, "emit_lag_median", 1, 2),
                     "90th percentile word delay (s)": f(r["emit_lag_p90"], 2),
                     "Recogniser calls per minute of audio": f(r["rec_calls_per_min"], 1),
                     "Real-time factor (95% CI)": withci(r, "rtf", 1, 2), "Engine": ENGINE[k]})
    T["rq2delay"] = md_table(pd.DataFrame(rows))

    T["vadgate"] = kv([
        ("Windows kept by the VAD gate", f"{V['p1v_windows_kept']}%"),
        ("Words where nobody spoke, removed", f"{V['p1v_red_nonspeech_tok_per_h']}% (Wilcoxon p = {V['p1v_p_nonspeech_tok_per_h']})"),
        ("WER, P1 and P1v", f"{V['wer_P1']}% and {V['wer_P1v']}% (Wilcoxon p = {V['p1v_p_wer']})"),
        ("Average WER reduction per excerpt, all 16 excerpts", f"{V['p1v_wer_red_pp']} percentage points"),
        ("Average WER reduction per excerpt, without the quietest excerpt", f"{V['p1v_wer_red_excl_pp']} percentage points"),
        ("Wrong fingerspelling under M0, removed", f"{V['p1v_red_spurious_fs_per_min']}% (Wilcoxon p = {V['p1v_p_spurious_fs_per_min']})"),
        (f"Quietest excerpt ({V['p1_worst_meeting']})",
         f"{V['p1_worst_refw']} reference words. P1 wrote {V['p1_worst_hypw']} words, WER {V['p1_worst_wer']}%. "
         f"P1v WER {V['p1v_worst_wer']}%. P2, P3 and P4 WER {V['others_worst_wer_lo']}% each"),
    ])

    def sign_rows(mapper, sched, spec):
        d = order(s[(s["mapper"] == mapper) & (s["schedule"] == sched)])
        rows = []
        for _, r in d.iterrows():
            row = {"Method": LABEL[r["policy"]]}
            for col, lab, sc, nd, ci_ in spec:
                v = f(r[col] * sc, nd)
                if ci_ and r["policy"] != "O-oracle":
                    v = withci(r, col, sc, nd)
                row[lab] = v
            rows.append(row)
        return rows

    T["rq3m0"] = md_table(pd.DataFrame(sign_rows("M0", "fifo@1x", [
        ("ser", "Sign error rate (%)", 100, 1, True), ("spurious_fs_per_min", "Wrong fingerspelled items per min", 1, 1, True),
        ("spurious_lex_per_min", "Wrong dictionary signs per min", 1, 1, False),
        ("fs_items_per_min_hyp", "Fingerspelled items per min", 1, 1, False),
        ("letters_per_min_hyp", "Letters per min", 1, 0, False), ("rho_hyp", "Signing demand at 1×", 1, 2, False)])))
    T["rq3m1"] = md_table(pd.DataFrame(sign_rows("M1", "fifo@1x", [
        ("ser", "Sign error rate (%)", 100, 1, True), ("spurious_fs_per_min", "Wrong fingerspelled items per min", 1, 1, True),
        ("spurious_lex_per_min", "Wrong dictionary signs per min", 1, 1, False),
        ("fs_items_per_min_hyp", "Fingerspelled items per min", 1, 1, False), ("rho_hyp", "Signing demand at 1×", 1, 2, False)])))

    rows = []
    for k, pid in (("P1", "P1-h1.0-c0.25"), ("P1v", "P1v-h1.0-c0.25-vad"), ("P2", "P2-sil500"), ("P3", "P3-la-1.0"), ("P4", "P4-aa-1.0")):
        rows.append({"Method": LABEL[pid], "Most frequent wrong items (count)": V[f"spur_top_{k}"],
                     "Share of THANK and YOU among wrong items (%)": V[f"thankyou_share_{k}"],
                     "Wrong fingerspelling per min: words where nobody spoke": V[f"causefs_nonspeech_{k}"],
                     "fragments of cut words": V[f"causefs_fragment_{k}"],
                     "other recognition errors": V[f"causefs_other_{k}"]})
    T["spurcauses"] = md_table(pd.DataFrame(rows))

    T["rq4m0"] = md_table(pd.DataFrame(sign_rows("M0", "fifo@1x", [
        ("rho_hyp", "Signing demand", 1, 2, False), ("sign_lag_median", "Median sign delay, s (95% CI)", 1, 1, True),
        ("sign_lag_p90", "90th percentile sign delay (s)", 1, 1, False),
        ("delivered_correct_lex_per_min", "Correct dictionary signs per min", 1, 1, False),
        ("overrun_s", "Signing still going after the excerpt ends (s)", 1, 0, False)])))

    d = order(s[(s["mapper"] == "M2") & (s["schedule"] == "bounded4@1.5x")])
    o_dlex = float(d[d["policy"] == "O-oracle"].iloc[0]["delivered_correct_lex_per_min"])
    rows = []
    for _, r in d.iterrows():
        rows.append({"Method": LABEL[r["policy"]],
                     "Correct dictionary signs per min (95% CI)": (withci(r, "delivered_correct_lex_per_min", 1, 1)
                                                                  if r["policy"] != "O-oracle" else f(r["delivered_correct_lex_per_min"], 1)),
                     "Relative to the perfect transcript (%)": f(100 * r["delivered_correct_lex_per_min"] / o_dlex, 0),
                     "Median sign delay (s)": f(r["sign_lag_median"], 2), "90th percentile sign delay (s)": f(r["sign_lag_p90"], 2),
                     "Items skipped by the lag limit (%)": f(100 * r["drop_rate"], 0),
                     "Wrong dictionary signs per min": f(r["spurious_lex_per_min"], 1)})
    T["rq4"] = md_table(pd.DataFrame(rows))

    rows = []
    for tag, lab, pk in (("m2l2", "2 s at 1.5×", "l2"), ("m2", "4 s at 1.5×", "l4"), ("m2l6", "6 s at 1.5×", "l6"), ("m2b1", "4 s at 1×", "b1")):
        rows.append({"Lag limit and speed": lab, "P1 fixed 1 s": V[f"dlex_{tag}_P1"], "P2 pause-based": V[f"dlex_{tag}_P2"],
                     "P3 LocalAgreement-2": V[f"dlex_{tag}_P3"], "P4 AlignAtt": V[f"dlex_{tag}_P4"],
                     "Perfect transcript": V[f"dlex_{tag}_O"], "P1 against P4, p": V[f"p_dlex_{pk}_P1P4"]})
    T["lagsens"] = md_table(pd.DataFrame(rows))

    T["correl"] = md_table(pd.DataFrame([
        {"Sign measure": "Sign error rate (M0)", "Transcript measure": "WER", "Spearman correlation": V["rsw_ser_wer"]},
        {"Sign measure": "Correct dictionary signs delivered (M2, lag limit 4 s, 1.5×)", "Transcript measure": "WER",
         "Spearman correlation": V["rsw_dlex_wer"]},
        {"Sign measure": "Correct dictionary signs delivered (M2, lag limit 4 s, 1.5×)", "Transcript measure": "median word delay",
         "Spearman correlation": V["rsw_dlex_lag"]},
        {"Sign measure": "Correct dictionary signs delivered (M2, lag limit 4 s, 1.5×)",
         "Transcript measure": "90th percentile word delay", "Spearman correlation": V["rsw_dlex_lag90"]},
    ]).replace(r"^-(\d)", "\u2212\\1", regex=True))

    # ---- statistics (comma-separated, no semicolons)
    st = pd.read_csv(os.path.join(STREAM, "stats_pairwise.csv"))
    st = st[(st["model"] == model) & (st["comp_aware"] == aware)]
    names = [("wer", "WER", "transcript"), ("emit_lag_median", "Median word delay", "transcript"),
             ("frag_per_1k", "Fragments", "transcript"), ("nonspeech_tok_per_h", "Words where nobody spoke", "transcript"),
             ("spurious_fs_per_min", "Wrong fingerspelled items", "M0, all in order, 1×"),
             ("spurious_lex_per_min", "Wrong dictionary signs", "M0, all in order, 1×"),
             ("rho_hyp", "Signing demand", "M0, all in order, 1×"),
             ("delivered_correct_lex_per_min", "Correct dictionary signs delivered", "M2, lag limit 4 s, 1.5×"),
             ("sign_lag_median", "Median sign delay", "M2, lag limit 4 s, 1.5×")]
    rows = []
    for col, lab, ctx in names:
        sel = st[st["metric"] == col]
        if ctx.startswith("M0"):
            sel = sel[(sel["mapper"] == "M0") & (sel["schedule"] == "fifo@1x")]
        elif ctx.startswith("M2"):
            sel = sel[(sel["mapper"] == "M2") & (sel["schedule"] == "bounded4@1.5x")]
        fr = sel.iloc[0]
        pairs = []
        for _, r in sel.iterrows():
            a, b = SHORT[r["a"]], SHORT[r["b"]]
            star = "**" if r["p_holm"] < 0.05 else ""
            pairs.append(f"{star}{a}–{b}: {pfmt(r['p_holm'])}{star}")
        rows.append({"Measure": lab, "Setting": ctx, "Friedman χ²(3), p": f"{f(fr['friedman_chi2'], 1)}, {pfmt(fr['friedman_p'])}",
                     "Pairwise Wilcoxon, Holm-adjusted p": ", ".join(pairs)})
    T["stats4"] = md_table(pd.DataFrame(rows))
    return T


def pfmt(p):
    """A p-value with two significant digits, small values as 4.5 × 10^−7^ (superscript) instead of 4.5e-07."""
    s = f"{p:.2g}"
    m = re.match(r"^(\d(?:\.\d+)?)e-0*(\d+)$", s)
    return f"{m.group(1)} × 10^−{m.group(2)}^" if m else s


def ore_layout(md):
    """Take the figures (image line and legend) and the tables (caption, table and note) out of the text, in order."""
    lines = md.split("\n")
    keep, figures, tables = [], [], []
    i = 0

    def next_nonblank(k):
        while k < len(lines) and not lines[k].strip():
            k += 1
        return k
    while i < len(lines):
        line = lines[i]
        if line.startswith("![](") and "<!--FIGURE_LEGENDS-->" in "\n".join(lines[i:]):
            k = next_nonblank(i + 1)
            if not re.match(r"^\*\*Figure \d+\. ", lines[k]):
                raise SystemExit(f"figure without legend: {line}")
            figures.append(line + "\n\n" + lines[k])
            i = k + 1
            continue
        m = re.match(r"^\*\*Table (\d+)\.\*\* ", line)
        if m and "<!--TABLES-->" in "\n".join(lines[i:]):
            k = next_nonblank(i + 1)
            rows = []
            while k < len(lines) and lines[k].startswith("|"):
                rows.append(lines[k])
                k += 1
            if len(rows) < 3:
                raise SystemExit(f"Table {m.group(1)}: no table after its caption")
            block = line + "\n\n" + "\n".join(rows)
            k2 = next_nonblank(k)
            if k2 < len(lines) and lines[k2].startswith("*Note:*"):
                block += "\n\n" + lines[k2]
                k = k2 + 1
            tables.append(block)
            i = k
            continue
        keep.append(line)
        i += 1
    for kind, items in (("Figure", figures), ("Table", tables)):
        nums = [int(re.search(rf"\*\*{kind} (\d+)\.", x).group(1)) for x in items]
        if nums != list(range(1, len(nums) + 1)):
            raise SystemExit(f"{kind}s out of order: {nums}")
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(keep))
    return text, figures, tables


def main():
    tmpl = open(os.path.join(HERE, "paper_template_v4.md"), encoding="utf-8").read()
    refs = {k: " ".join(v.split()) for k, v in yaml.safe_load(open(os.path.join(HERE, "references_v4.yaml"))).items()}
    V, _ = bp.oracle_vars()
    V2, _ = bp.stream_tables()
    V.update(V2)
    T = tables(V)
    out = bp.render(tmpl, V, T, refs)
    # Open Research Europe asks for the figure legends and the tables at the end of the manuscript, not in the text:
    # the figures (image and legend) and the tables (caption, table and note) move to the last two sections
    out, figures, table_blocks = ore_layout(out)
    out = out.replace("<!--FIGURE_LEGENDS-->", "\n\n".join(figures)).replace("<!--TABLES-->", "\n\n".join(table_blocks))
    legends = [f.split("\n\n", 1)[1] for f in figures]
    for lg in legends:
        title = re.match(r"\*\*Figure \d+\. (.*?)\*\*", lg).group(1)
        if len(title.split()) > 15:
            print("FIGURE TITLE LONGER THAN 15 WORDS:", title)
    abstract = out.split("# Abstract", 1)[1].split("# Keywords", 1)[0]
    keywords = out.split("# Keywords", 1)[1].split("\n# ", 1)[0].strip().split(", ")
    first_person = re.findall(r"\b(we|our|ours|us)\b", out, flags=re.I)
    print(f"abstract: {len(abstract.replace('**', '').split())} words with headings, "
          f"headings {re.findall(r'^[*][*](\w+)[*][*]$', abstract, flags=re.M)}, {len(keywords)} keywords, "
          f"{len(legends)} figure legends at the end, first person: {len(first_person)}")
    dst = os.path.join(HERE, "INTERACT_StreamingS2S_ORE_v5.md")
    open(dst, "w", encoding="utf-8").write(out)
    n_pending = len(re.findall(r"\[PENDING:", out))
    semis = [ln for ln in out.splitlines() if ";" in ln]
    print(f"wrote {dst}  ({len(out.split()):,} words, {n_pending} pending items, {len(semis)} lines with a semicolon)")
    for ln in semis[:10]:
        print("  SEMICOLON:", ln[:160])
    unused = sorted(set(refs) - set(re.findall(r"@([A-Za-z0-9_]+)", tmpl)))
    if unused:
        print("references not cited:", ", ".join(unused))


if __name__ == "__main__":
    sys.exit(main())
