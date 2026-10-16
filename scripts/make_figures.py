#!/usr/bin/env python3
"""Figures for the paper (300 dpi PNG + vector PDF).

    python scripts/make_figures.py --oracle results/oracle --streaming results/streaming --out figures

Oracle figures (RQ1) need only results/oracle; streaming figures are produced
when results/streaming exists.  Colours: validated categorical palette
(slots 1-4: blue, orange, aqua, yellow), every series also direct-labelled.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MAPPER_COLOR = {"M0": SLOTS[0], "M1": SLOTS[1], "M2": SLOTS[2]}
MAPPER_LABEL = {"M0": "M0 deployed (word lookup + fingerspelling)",
                "M1": "M1 load-reduced (+ lemmas, phrases, function-word omission)",
                "M2": "M2 load-reduced, no fingerspelling (OOV to captions)"}
POLICY_COLOR = {"P1": SLOTS[0], "P2": SLOTS[1], "P3": SLOTS[2], "P4": SLOTS[3]}
POLICY_NAME = {"P1": "P1 fixed chunks", "P2": "P2 pause/VAD", "P3": "P3 LocalAgreement-2", "P4": "P4 AlignAtt"}
POLICY_MARKER = {"P1": "o", "P2": "s", "P3": "^", "P4": "D"}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "savefig.dpi": 300, "savefig.bbox": "tight", "legend.frameon": False,
    })


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print("wrote", os.path.join(out, name + ".png"))


# --------------------------------------------------------------- Figure 1 --
def fig_pipeline(out):
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.set_axis_off()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 34)
    W, X0, GAP = 16.0, 1.0, 4.0
    boxes = [
        ("Meeting audio", "AMI Mix-Headset\n16 kHz, real time", "#f0efec", ""),
        ("Streaming ASR\npolicy", "P1 fixed · P2 pause/VAD\nP3 LocalAgreement-2\nP4 AlignAtt",
         "#cde2fb", "WER, emission lag,\nfragments, RTF"),
        ("Committed\nwords", "text, ASR times,\nemission time", "#f0efec", ""),
        ("Text-to-sign\nmapper", "M0 · M1 · M2\n744-key IS lexicon", "#cde2fb", "sign error rate,\nfingerspelling load"),
        ("Avatar\nplayback", "sequential clips,\nFIFO or bounded lag", "#cde2fb", "sign lag, demand ρ,\ndelivered signs"),
    ]
    for k, (title, detail, fc, metric) in enumerate(boxes):
        x = X0 + k * (W + GAP)
        ax.add_patch(FancyBboxPatch((x, 19), W, 10, boxstyle="round,pad=0.3,rounding_size=1.2",
                                    fc=fc, ec=AXIS, lw=0.8))
        ax.text(x + W / 2, 24, title, ha="center", va="center", fontsize=8, color=INK, linespacing=1.15)
        ax.text(x + W / 2, 13.2, detail, ha="center", va="center", fontsize=6.6, color=INK2, linespacing=1.2)
        if metric:
            ax.text(x + W / 2, 3.6, metric, ha="center", va="center", fontsize=6.6, color=MUTED,
                    linespacing=1.2, style="italic")
        if k < len(boxes) - 1:
            ax.add_patch(FancyArrowPatch((x + W + 0.5, 24), (x + W + GAP - 0.5, 24), arrowstyle="-|>",
                                         mutation_scale=8, color=INK2, lw=0.9))
    ax.text(0.5, 32.6, "Oracle path (RQ1): AMI manual transcript → same mapper → playback with zero ASR delay",
            ha="left", va="center", fontsize=6.8, color=MUTED)
    save(fig, out, "fig1_pipeline")


# ------------------------------------------------------------ oracle figs --
def fig_demand(oracle_dir, out):
    s = pd.read_csv(os.path.join(oracle_dir, "oracle_summary.csv"))
    s = s[(s["schedule"] == "fifo") & (s["asr_delay"] == 0)]
    rates = sorted(s["rate"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, mode, title in zip(axes, ["active", "full"],
                               ["(a) clips trimmed to the hand-visible span", "(b) full clips (all frames)"]):
        d = s[s["mode"] == mode]
        w = 0.24
        x = np.arange(len(rates))
        for k, m in enumerate(["M0", "M1", "M2"]):
            dm = d[d["mapper"] == m].set_index("rate").loc[rates]
            xs = x + (k - 1) * w
            ax.bar(xs, dm["rho"], width=w, color=MAPPER_COLOR[m], edgecolor="white", linewidth=1.0,
                   label=m, zorder=3)
            ax.errorbar(xs, dm["rho"], yerr=[dm["rho"] - dm["rho_lo"], dm["rho_hi"] - dm["rho"]],
                        fmt="none", ecolor=INK2, elinewidth=0.7, capsize=0, zorder=4)
        ax.axhline(1.0, color=INK, lw=0.9, zorder=5)
        ax.text(-1.02, 1.07, "ρ = 1", ha="left", va="bottom", fontsize=7, color=INK)
        ax.set_xlim(-1.08, len(rates) - 0.55)
        ax.set_yscale("log")
        ax.set_yticks([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64])
        ax.set_yticklabels(["0.25", "0.5", "1", "2", "4", "8", "16", "32", "64"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:g}×" for r in rates])
        ax.set_xlabel("playback rate of the 30 fps clips")
        ax.set_title(title, loc="left", color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("signing demand ρ\n(s of signing per s of meeting)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, [MAPPER_LABEL[x] for x in l], loc="upper center", ncol=1, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "fig2_signing_demand")


def fig_bounded(oracle_dir, out):
    s = pd.read_csv(os.path.join(oracle_dir, "oracle_summary.csv"))
    s = s[(s["mode"] == "active") & (s["asr_delay"] == 0) & (s["schedule"].str.startswith("bounded"))]
    s = s.assign(maxlag=s["schedule"].str.replace("bounded", "").astype(float))
    fifo = pd.read_csv(os.path.join(oracle_dir, "oracle_summary.csv"))
    fifo = fifo[(fifo["mode"] == "active") & (fifo["schedule"] == "fifo") & (fifo["asr_delay"] == 0) & (fifo["rate"] == 1.0)]
    ceil = fifo.set_index("mapper")["lex_per_min"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, rate in zip(axes, [1.0, 1.5]):
        d = s[s["rate"] == rate]
        for m, lab in (("M2", "all lexical signs, M1/M2"), ("M0", "all lexical signs, M0")):
            ax.axhline(ceil[m], color=AXIS, lw=0.9, zorder=1)
            ax.text(1.75, ceil[m] - 0.8, lab, fontsize=6.6, color=MUTED, va="top", ha="left", zorder=5)
        for m in ["M0", "M1", "M2"]:
            dm = d[d["mapper"] == m].sort_values("maxlag")
            ax.plot(dm["maxlag"], dm["delivered_lex_per_min"], color=MAPPER_COLOR[m], lw=1.6, marker="o",
                    ms=5, mec="white", mew=1.2, zorder=3, label=m)
            ax.fill_between(dm["maxlag"], dm["delivered_lex_per_min_lo"], dm["delivered_lex_per_min_hi"],
                            color=MAPPER_COLOR[m], alpha=0.10, lw=0, zorder=2)
            last = dm.iloc[-1]
            ax.text(last["maxlag"] + 0.12, last["delivered_lex_per_min"], m, va="center", fontsize=7.5, color=INK)
        ax.set_xticks([2, 4, 6])
        ax.set_xticklabels(["2 s", "4 s", "6 s"])
        ax.set_xlim(1.7, 6.6)
        ax.set_xlabel("maximum allowed sign lag (bounded-lag playback)")
        ax.set_title(f"({'a' if rate == 1.0 else 'b'}) playback rate {rate:g}×", loc="left", color=INK)
    axes[0].set_ylabel("lexical signs delivered\nper minute")
    axes[0].set_ylim(0, 62)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, [MAPPER_LABEL[x] for x in l], loc="upper center", ncol=1, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "fig3_bounded_lag_delivery")


# --------------------------------------------------------- streaming figs --
O_COLOR = "#898781"   # reference row (perfect transcript): neutral, never a series hue
SHORT = {"P1-h1.0-c0.25": "P1", "P1v-h1.0-c0.25-vad": "P1v", "P2-sil500": "P2", "P3-la-1.0": "P3",
         "P4-aa-1.0": "P4", "O-oracle": "O"}
LONG = {"P1": "P1 fixed 1 s chunks (deployed)", "P1v": "P1v = P1 + VAD gate", "P2": "P2 pause/VAD utterances",
        "P3": "P3 LocalAgreement-2", "P4": "P4 AlignAtt", "O": "O perfect transcript"}


def _family(pid):
    return pid.split("-")[0]


def _select(df):
    """The streaming experiment of the paper: the most frequent model and simulation mode."""
    model = df["model"].mode().iloc[0]
    aware = bool(df["comp_aware"].mode().iloc[0])
    return df[(df["model"] == model) & (df["comp_aware"] == aware)], model, aware


def _ordered(df):
    df = df.copy()
    df["_o"] = df["policy"].map({p: i for i, p in enumerate(SHORT)}).fillna(99)
    return df.sort_values("_o").drop(columns="_o")


def fig_tradeoff(stream_dir, out):
    t, model, aware = _select(pd.read_csv(os.path.join(stream_dir, "summary_transcript.csv")))
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for _, r in _ordered(t).iterrows():
        short = SHORT.get(r["policy"], r["policy"])
        fam = short[:2]
        variant = short == "P1v"
        ax.errorbar(r["emit_lag_median"], 100 * r["wer"],
                    xerr=[[r["emit_lag_median"] - r["emit_lag_median_lo"]], [r["emit_lag_median_hi"] - r["emit_lag_median"]]],
                    yerr=[[100 * (r["wer"] - r["wer_lo"])], [100 * (r["wer_hi"] - r["wer"])]],
                    fmt=POLICY_MARKER[fam], color=POLICY_COLOR[fam], ms=7, mec=POLICY_COLOR[fam] if variant else "white",
                    mfc="white" if variant else POLICY_COLOR[fam], mew=1.2,
                    ecolor=POLICY_COLOR[fam], elinewidth=0.8, zorder=3)
        ax.annotate(short, (r["emit_lag_median"], 100 * r["wer"]), xytext=(6, 5 if not variant else -11),
                    textcoords="offset points", fontsize=8, color=INK)
    ax.set_xlabel("median emission lag (s), computation-unaware" if not aware else "median emission lag (s)")
    ax.set_ylabel("WER (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    names = [p for p in POLICY_NAME]
    handles = [plt.Line2D([], [], ls="", marker=POLICY_MARKER[p], color=POLICY_COLOR[p], ms=6, mec="white")
               for p in names]
    labels = [LONG[p] for p in names]
    if (t["policy"] == "P1v-h1.0-c0.25-vad").any():
        handles.insert(1, plt.Line2D([], [], ls="", marker="o", color=POLICY_COLOR["P1"], mfc="white", ms=6))
        labels.insert(1, LONG["P1v"])
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "fig5_latency_quality")


def _color(f_):
    return O_COLOR if f_ == "O" else POLICY_COLOR[f_[:2]]


def _hbar(ax, d, col, scale=1.0, nd=1, unit=""):
    y = np.arange(len(d))
    fams = [SHORT.get(p, p) for p in d["policy"]]
    cols = [_color(f_) for f_ in fams]
    v = scale * d[col].values
    bars = ax.barh(y, v, height=0.58, color=cols, edgecolor="white", linewidth=1.0, zorder=3)
    for b, f_ in zip(bars, fams):
        if f_ == "P1v":   # variant of P1: same hue, hatched
            b.set_facecolor("white")
            b.set_edgecolor(POLICY_COLOR["P1"])
            b.set_hatch("////")
            b.set_linewidth(0.8)
    has_ci = col + "_lo" in d
    if has_ci:
        lo, hi = scale * d[col + "_lo"].values, scale * d[col + "_hi"].values
        m = np.array([f_ != "O" for f_ in fams])
        ax.errorbar(v[m], y[m], xerr=[v[m] - lo[m], hi[m] - v[m]], fmt="none", ecolor=INK2, elinewidth=0.7, zorder=4)
        ends = np.where(m, hi, v)
    else:
        ends = v
    for yi, e, val in zip(y, ends, v):
        ax.text(e, yi, f"  {val:.{nd}f}{unit}", va="center", fontsize=7, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([LONG.get(f_, f_) for f_ in fams])
    if not ax.yaxis_inverted():   # shared y axes: invert only once
        ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(ends) * 1.25 if len(ends) else 1)


def fig_spurious(stream_dir, out):
    s, model, aware = _select(pd.read_csv(os.path.join(stream_dir, "summary_sign.csv")))
    d = _ordered(s[(s["schedule"] == "fifo@1x") & (s["mapper"] == "M0")])
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    _hbar(axes[0], d, "spurious_fs_per_min", nd=1)
    axes[0].set_xlabel("spurious fingerspelled items per minute")
    axes[0].set_title("(a) spurious fingerspelling", loc="left", color=INK, fontsize=8)
    _hbar(axes[1], d, "rho_hyp", nd=2)
    axes[1].set_xlabel("signing demand ρ at 1× (s of signing per s of audio)")
    axes[1].set_title("(b) signing demand", loc="left", color=INK, fontsize=8)
    save(fig, out, "fig6_spurious_fingerspelling")


def fig_sign_lag(stream_dir, out, mapper="M2", schedule="bounded4@1.5x"):
    s, model, aware = _select(pd.read_csv(os.path.join(stream_dir, "summary_sign.csv")))
    d = _ordered(s[(s["mapper"] == mapper) & (s["schedule"] == schedule)])
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    _hbar(axes[0], d, "delivered_correct_lex_per_min", nd=1)
    axes[0].set_xlabel("correct lexical signs delivered per minute")
    axes[0].set_title("(a) delivered signs", loc="left", color=INK, fontsize=8)
    _hbar(axes[1], d, "sign_lag_median", nd=2, unit=" s")
    axes[1].set_xlabel("median sign lag of delivered signs (s)")
    axes[1].set_title("(b) sign lag", loc="left", color=INK, fontsize=8)
    save(fig, out, "fig7_sign_lag_delivery")


class _LabelLevels:
    """Greedy vertical levels for rotated labels: a label goes to the lowest level whose last label is at
    least ``min_gap`` (data units) to its left."""

    def __init__(self, min_gap, levels=2):
        self.gap, self.last = min_gap, [-1e9] * levels

    def peek(self, x):
        for i, l in enumerate(self.last):
            if x - l >= self.gap:
                return i
        return int(np.argmin(self.last))

    def take(self, x, lev):
        self.last[lev] = x

    def level(self, x):
        lev = self.peek(x)
        self.take(x, lev)
        return lev


def fig_example(runs_dir, ref_dir, lexicon, out, meeting, t0, t1, model="small"):
    """Committed words of each policy over a short stretch of one excerpt.  Every commit is a tick at its
    emission time; the words of one commit fan out to its right, coloured by what the deployed mapper M0
    makes of them (sign or fingerspelling).  The top row shows the reference words at their end times."""
    import glob
    import sys
    from itertools import groupby

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from s2sbench.io import load_reference, load_run
    from s2sbench.lexicon import Lexicon
    from s2sbench.mapper import make_mapper
    from s2sbench.metrics import prepare_for_mapping

    lx = Lexicon.from_json(lexicon)
    m0 = make_mapper("M0", lx)

    def kind(w):
        items = m0.map([w], batch_by_emit=False)
        return items[0].kind if items else "LEX"

    tracks = []
    start = limit = None
    for pid in ["P1-h1.0-c0.25", "P2-sil500", "P3-la-1.0", "P4-aa-1.0"]:
        fs = glob.glob(os.path.join(runs_dir, model, pid + "*", f"{meeting}.json"))
        if not fs:
            return
        run = load_run(fs[0])
        start, limit = run.get("start_seconds", 0.0), run.get("limit_seconds")
        tracks.append((SHORT[pid], [w for w in prepare_for_mapping(run["words"]) if t0 <= w.emit <= t1]))
    ref, _ = load_reference(os.path.join(ref_dir, f"{meeting}.json"), limit=limit, start=start or 0.0)
    ref = sorted([w for w in ref if t0 <= w.end <= t1], key=lambda w: w.end)
    lex_c, fs_c = SLOTS[0], SLOTS[1]
    n = len(tracks) + 1
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    step, fs = 0.13, 5.6
    # reference row
    y = n - 1
    ax.axhline(y, color=GRID, lw=0.6, zorder=1)
    placer = _LabelLevels(min_gap=0.17, levels=3)
    for w in ref:
        c = lex_c if kind(w) == "LEX" else fs_c
        lev = placer.level(w.end)
        ax.plot([w.end, w.end], [y, y + 0.06 + 0.5 * lev], color=c, lw=0.6 if lev else 0.9, zorder=3)
        ax.text(w.end, y + 0.09 + 0.5 * lev, w.text.strip(), rotation=90, fontsize=fs, ha="center", va="bottom",
                color=c)
    ax.text(t0 - 0.15, y, "Speech\n(reference words\nat their end)", ha="right", va="center", fontsize=7, color=INK)
    # policy rows
    for row, (name, words) in enumerate(tracks):
        y = n - 2 - row
        ax.axhline(y, color=GRID, lw=0.6, zorder=1)
        placer = _LabelLevels(min_gap=step * 0.95)
        for emit, g in groupby(words, key=lambda w: w.emit):
            g = list(g)
            xs = [emit + step * k for k in range(len(g))]
            lev = max(placer.peek(x) for x in xs)   # one level for the whole commit
            for x in xs:
                placer.take(x, lev)
            yl = y + 0.5 * lev
            ax.plot([emit, emit], [y - 0.07, y + 0.07], color=INK, lw=1.1, zorder=4)
            if lev:
                ax.plot([emit, emit], [y + 0.07, yl], color=INK2, lw=0.5, zorder=3)
            if len(g) > 1:
                ax.plot([emit, xs[-1]], [yl, yl], color=INK2, lw=0.6, zorder=3)
            for x, w in zip(xs, g):
                c = lex_c if kind(w) == "LEX" else fs_c
                ax.text(x, yl + 0.09, w.text.strip(), rotation=90, fontsize=fs, ha="center", va="bottom", color=c)
        ax.text(t0 - 0.15, y, LONG[name].replace(" (deployed)", "\n(deployed)"), ha="right", va="center",
                fontsize=7, color=INK)
    ax.set_yticks([])
    right = t1 + 0.6
    for _, words in tracks:
        for emit, g in groupby(words, key=lambda w: w.emit):
            right = max(right, emit + step * (len(list(g)) - 1) + 0.25)
    ax.set_xlim(t0, right)
    ax.set_ylim(-0.25, n + 0.45)
    ax.set_xlabel(f"time (s) within the excerpt of AMI {meeting} (excerpt starts {start:.0f} s into the meeting)")
    ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], ls="", marker="s", ms=6, color=lex_c),
               plt.Line2D([], [], ls="", marker="s", ms=6, color=fs_c),
               plt.Line2D([], [], ls="-", lw=1.1, color=INK)]
    fig.legend(handles, ["word with a lexicon sign (M0)", "word fingerspelled by M0",
                         "commit (emission time); its words fan out to the right"],
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.0))
    save(fig, out, "fig4_example_commits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="results/oracle")
    ap.add_argument("--streaming", default="results/streaming")
    ap.add_argument("--runs", default="runs_cpu")
    ap.add_argument("--ref", default="data/ami/ref")
    ap.add_argument("--lexicon", default="data/lexicon.json")
    ap.add_argument("--example", nargs=3, default=None, metavar=("MEETING", "T0", "T1"),
                    help="excerpt and time window (s) for the example figure, e.g. ES2004b 60 75")
    ap.add_argument("--out", default="manuscript/figures")
    a = ap.parse_args()
    style()
    fig_pipeline(a.out)
    if os.path.exists(os.path.join(a.oracle, "oracle_summary.csv")):
        fig_demand(a.oracle, a.out)
        fig_bounded(a.oracle, a.out)
    if os.path.exists(os.path.join(a.streaming, "summary_transcript.csv")):
        fig_tradeoff(a.streaming, a.out)
        fig_spurious(a.streaming, a.out)
        fig_sign_lag(a.streaming, a.out)
    if a.example:
        fig_example(a.runs, a.ref, a.lexicon, a.out, a.example[0], float(a.example[1]), float(a.example[2]))


if __name__ == "__main__":
    main()
