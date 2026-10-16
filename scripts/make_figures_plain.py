#!/usr/bin/env python3
"""Figures for version 4 of the paper, prepared to the figure guidelines of Open Research Europe.

    python scripts/make_figures_plain.py --example ES2004a 35 46.5

Open Research Europe asks that line art (graphs and diagrams) uses RGB colour, a white background, text of
at least 8 pt in a consistent typeface and lines that stay visible at the printed width of 75 mm (one column)
or 150 mm (two columns), and that graphs are supplied as EPS files exported from the program that made them,
or else as uncompressed TIFF files of at least 600 dpi at the printed size.  Every figure is therefore drawn at
its final size, 150 mm or 75 mm wide, with all text at 8 pt in Liberation Sans (metrically identical to Arial),
without transparency, and written as

  * <name>.eps and <name>.pdf   vector files, fonts embedded as TrueType (Type 42),
  * <name>.tif                  RGB, uncompressed, 600 dpi at the printed size, sRGB profile embedded,
  * <name>.png                  RGB, 300 dpi at the printed size, sRGB profile embedded (Word and Markdown files).

figure_check.md, written next to the figures, lists the measured size, resolution, colour mode, compression,
embedded fonts and smallest text of every figure, and any text that is clipped or overlaps other text.

Oracle figures (RQ1) need only results/oracle; streaming figures are produced when results/streaming exists.
Colours: validated categorical palette (slots 1-4: blue, orange, aqua, yellow), every series also labelled.
"""
import argparse
import glob
import io
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402
from matplotlib.font_manager import FontProperties, findfont  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.textpath import TextToPath  # noqa: E402
from PIL import Image, ImageCms, TiffImagePlugin  # noqa: E402
from PIL.PngImagePlugin import PngInfo  # noqa: E402

# ---------------------------------------------------------------- page geometry and typography --
PX_FULL, PX_ONE = 1770, 886          # width in pixels at 300 dpi: 149.9 mm (two columns) and 75.0 mm (one column)
FONT = "Liberation Sans"             # metrically identical to Arial
FS = 8.0                             # size of every text in every figure (pt)
AUTHORS = "N. D. Tantaroudas, A. J. McCracken, I. Karachalios"
CREATOR = f"matplotlib {matplotlib.__version__} (scripts/make_figures_plain.py)"
SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()

INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MAPPER_COLOR = {"M0": SLOTS[0], "M1": SLOTS[1], "M2": SLOTS[2]}
MAPPER_LABEL = {"M0": "M0 deployed (dictionary lookup, other words fingerspelled)",
                "M1": "M1 reduced (word forms and phrases, small words skipped)",
                "M2": "M2 reduced, no fingerspelling (unknown words to captions)"}
POLICY_COLOR = {"P1": SLOTS[0], "P2": SLOTS[1], "P3": SLOTS[2], "P4": SLOTS[3]}
POLICY_NAME = {"P1": "P1 fixed chunks", "P2": "P2 pause-based", "P3": "P3 LocalAgreement-2", "P4": "P4 AlignAtt"}
POLICY_MARKER = {"P1": "o", "P2": "s", "P3": "^", "P4": "D"}
# word colours of Figure 4: the blue and orange slots darkened to a text contrast of at least 4.5:1 on white
SIGN_TXT, FS_TXT = "#266cc1", "#bc532a"


def blend(color, a):
    """Opaque colour that looks like ``color`` at opacity ``a`` on white (EPS has no transparency)."""
    return to_hex(tuple(1 - a + a * c for c in to_rgb(color)))


def fsize(px300, h_mm):
    """Figure size in inches for a width of ``px300`` pixels at 300 dpi and a height in mm (whole pixels)."""
    return (px300 + 0.01) / 300, (round(h_mm / 25.4 * 300) + 0.01) / 300


def style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [FONT, "Arial", "DejaVu Sans"],
        "font.size": FS, "axes.titlesize": FS, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 4, "axes.labelsize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
        "legend.fontsize": FS, "legend.title_fontsize": FS, "figure.titlesize": FS,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.major.pad": 2.5, "ytick.major.pad": 2.5,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "savefig.edgecolor": "white", "savefig.transparent": False, "savefig.bbox": None,
        "legend.frameon": False, "legend.framealpha": 1.0, "legend.handlelength": 1.6,
        "legend.columnspacing": 1.6, "legend.borderaxespad": 0.2, "legend.borderpad": 0.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "hatch.linewidth": 0.8,
        "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
    })


# ---------------------------------------------------------------------------- saving and checks --
def _audit(fig, name, width_label):
    """Sizes and positions of all visible texts: smallest size, typefaces, clipped and overlapping texts."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    hidden = set()      # tick labels of ticks outside the axis range exist but are not drawn
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lo, hi = sorted(axis.get_view_interval())
            eps = 1e-9 * max(1.0, abs(hi - lo))
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not (lo - eps <= tick.get_loc() <= hi + eps):
                    hidden.update({id(tick.label1), id(tick.label2)})
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip() and id(t) not in hidden]
    boxes = [t.get_window_extent(r) for t in texts]
    tol = 0.5
    clipped = [t.get_text() for t, b in zip(texts, boxes)
               if b.x0 < fb.x0 - tol or b.x1 > fb.x1 + tol or b.y0 < fb.y0 - tol or b.y1 > fb.y1 + tol]
    overlaps = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            w = min(a.x1, b.x1) - max(a.x0, b.x0)
            h = min(a.y1, b.y1) - max(a.y0, b.y0)
            if w > 0 and h > 0 and w * h > 0.12 * min(a.width * a.height, b.width * b.height):
                overlaps.append((texts[i].get_text(), texts[j].get_text()))
    fonts = sorted({os.path.basename(findfont(t.get_fontproperties())) for t in texts})
    return {"name": name, "width": width_label, "n_text": len(texts),
            "min_pt": min(t.get_fontsize() for t in texts), "max_pt": max(t.get_fontsize() for t in texts),
            "fonts": fonts, "clipped": clipped, "overlaps": overlaps}


def save(fig, out, name, title, check, width_label="two columns"):
    os.makedirs(out, exist_ok=True)
    check.append(_audit(fig, name, width_label))
    base = os.path.join(out, name)
    fig.savefig(base + ".pdf", metadata={"Title": title, "Author": AUTHORS, "Creator": CREATOR})
    fig.savefig(base + ".eps", metadata={"Creator": CREATOR})
    for dpi, ext in ((300, "png"), (600, "tif")):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        im = Image.open(buf).convert("RGBA")
        rgb = Image.alpha_composite(Image.new("RGBA", im.size, (255, 255, 255, 255)), im).convert("RGB")
        if ext == "png":
            info = PngInfo()
            for k, v in (("Title", title), ("Author", AUTHORS), ("Software", CREATOR)):
                info.add_text(k, v)
            rgb.save(base + ".png", dpi=(dpi, dpi), icc_profile=SRGB, pnginfo=info, optimize=True)
        else:
            tags = TiffImagePlugin.ImageFileDirectory_v2()
            tags[270], tags[315], tags[305] = title, AUTHORS, CREATOR    # description, artist, software
            rgb.save(base + ".tif", dpi=(dpi, dpi), icc_profile=SRGB, compression=None, tiffinfo=tags)
    plt.close(fig)
    print("wrote", base + ".{eps,pdf,tif,png}")


def write_check(out, check):
    """figure_check.md: what the files contain, measured from the files themselves."""
    rows = []
    for c in check:
        base = os.path.join(out, c["name"])
        png, tif = Image.open(base + ".png"), Image.open(base + ".tif")
        eps = open(base + ".eps", "rb").read().decode("latin-1")
        bb = [float(v) for v in re.search(r"%%BoundingBox:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", eps).groups()]
        pdf = open(base + ".pdf", "rb").read()
        mb = [float(v) for v in re.search(rb"/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", pdf).groups()]
        comp = {1: "none"}.get(tif.tag_v2.get(259), str(tif.tag_v2.get(259)))
        png_dpi, tif_dpi = float(png.info["dpi"][0]), float(tif.info["dpi"][0])
        rows.append({
            "Figure": c["name"], "Printed width": c["width"],
            "Size (mm)": f"{25.4 * png.width / png_dpi:.1f} × {25.4 * png.height / png_dpi:.1f}",
            "PNG": f"{png.width} × {png.height} px, {png_dpi:.0f} dpi, {png.mode}, sRGB {'yes' if png.info.get('icc_profile') else 'no'}",
            "TIFF": f"{tif.width} × {tif.height} px, {tif_dpi:.0f} dpi, {tif.mode}, compression {comp}, "
                    f"sRGB {'yes' if tif.info.get('icc_profile') else 'no'}",
            "EPS and PDF (mm)": f"{25.4 * (bb[2] - bb[0]) / 72:.1f} × {25.4 * (bb[3] - bb[1]) / 72:.1f} and "
                                f"{25.4 * (mb[2] - mb[0]) / 72:.1f} × {25.4 * (mb[3] - mb[1]) / 72:.1f}",
            "Fonts embedded": ("TrueType" if ("/FontType 42" in eps and b"/FontFile2" in pdf) else "check"),
            "Text": f"{c['n_text']} items, {c['min_pt']:g} to {c['max_pt']:g} pt, {', '.join(c['fonts'])}",
            "Clipped or overlapping text": (", ".join(c["clipped"] + [f"{a} / {b}" for a, b in c["overlaps"]]) or "none"),
        })
    cols = list(rows[0])
    lines = ["# Figure check", "",
             "Measured from the figure files by scripts/make_figures_plain.py. Open Research Europe asks for RGB "
             "colour, a white background, text of at least 8 pt in a consistent typeface, a printed width between "
             "75 mm (one column) and 150 mm (two columns), and EPS files for graphs or else uncompressed TIFF files "
             "of at least 600 dpi at the printed size.", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r[k]) for k in cols) + " |" for r in rows]
    open(os.path.join(out, "figure_check.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    bad = [c["name"] for c in check if c["min_pt"] < 8 or c["clipped"] or c["overlaps"]]
    print("figure check:", "all figures pass" if not bad else "CHECK " + ", ".join(bad))


# ------------------------------------------------------------------------------------ Figure 1 --
CAP, LINE = 2.02, 3.4      # cap height and line spacing of 8 pt text (mm)


def _block(ax, x, y_c, text, **kw):
    """Lines of text centred on y_c (mm) with a fixed line spacing."""
    lines = text.split("\n")
    b0 = y_c - (CAP - (len(lines) - 1) * LINE) / 2
    for i, ln in enumerate(lines):
        ax.text(x, b0 - i * LINE, ln, va="baseline", **kw)


def _lines(ax, x, y_top, text, **kw):
    """Lines of text hanging from y_top (mm) with a fixed line spacing."""
    for i, ln in enumerate(text.split("\n")):
        ax.text(x, y_top - 2.4 - i * LINE, ln, va="baseline", **kw)


def fig_pipeline(out, check):
    W, H = PX_FULL / 300 * 25.4, 51.0
    fig = plt.figure(figsize=fsize(PX_FULL, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    bw, m = 25.0, 0.6
    gap = (W - 2 * m - 5 * bw) / 4
    y0, bh = 28.0, 12.0
    light, blue = "#f0efec", "#cde2fb"
    stages = [
        ("Meeting audio", "AMI headset mix\n16 kHz, real time", light, ""),
        ("Streaming\nrecognition", "P1 fixed chunks\nP2 pause-based\nP3 LocalAgreement-2\nP4 AlignAtt", blue,
         "WER, word delay,\nfragments,\ncomputing cost"),
        ("Released\nwords", "text, word times\nand release time", light, ""),
        ("Text-to-sign\nmapping", "rules M0, M1, M2\nIS dictionary\nof 744 entries", blue,
         "sign error rate,\nfingerspelling load"),
        ("Avatar\nplayback", "one clip at a time,\nall in order or\nwith a lag limit", blue,
         "sign delay, demand,\ndelivered signs"),
    ]
    xs = [m + k * (bw + gap) for k in range(len(stages))]
    for k, (title, detail, fc, metric) in enumerate(stages):
        x = xs[k]
        ax.add_patch(FancyBboxPatch((x, y0), bw, bh, boxstyle="round,pad=0,rounding_size=1.6",
                                    fc=fc, ec=AXIS, lw=0.8))
        _block(ax, x + bw / 2, y0 + bh / 2, title, ha="center", fontsize=FS, fontweight="bold", color=INK)
        _lines(ax, x + bw / 2, y0 - 1.6, detail, ha="center", fontsize=FS, color=INK2)
        if metric:
            _lines(ax, x + bw / 2, 11.4, metric, ha="center", fontsize=FS, color=INK2, style="italic")
        if k < len(stages) - 1:
            ax.add_patch(FancyArrowPatch((x + bw + 0.7, y0 + bh / 2), (xs[k + 1] - 0.7, y0 + bh / 2),
                                         arrowstyle="-|>", mutation_scale=7, color=INK2, lw=0.9))
    # perfect-transcript path: from the meeting straight to the released words, bypassing the recogniser
    c0, c2, top, yb = xs[0] + bw / 2, xs[2] + bw / 2, y0 + bh, y0 + bh + 4.6
    ax.plot([c0, c0, c2], [top + 0.3, yb, yb], color=INK2, lw=0.9, ls=(0, (3, 2)), solid_capstyle="butt")
    ax.add_patch(FancyArrowPatch((c2, yb), (c2, top + 0.4), arrowstyle="-|>", mutation_scale=7, color=INK2,
                                 lw=0.9, linestyle=(0, (3, 2))))
    ax.text((c0 + c2) / 2, yb + 1.3, "perfect transcript (RQ1), no recognition delay", ha="center",
            va="baseline", fontsize=FS, color=INK2)
    save(fig, out, "fig1_pipeline", "Figure 1. The benchmark pipeline", check)


# --------------------------------------------------------------------------------- oracle figures --
def fig_demand(oracle_dir, out, check):
    s = pd.read_csv(os.path.join(oracle_dir, "oracle_summary.csv"))
    s = s[(s["schedule"] == "fifo") & (s["asr_delay"] == 0)]
    rates = sorted(s["rate"].unique())
    fig, axes = plt.subplots(1, 2, figsize=fsize(PX_FULL, 80), sharey=True, layout="constrained")
    for ax, mode, title in zip(axes, ["active", "full"],
                               ["(a) clips trimmed to visible hands", "(b) full clips (all frames)"]):
        d = s[s["mode"] == mode]
        w = 0.24
        x = np.arange(len(rates))
        for k, m in enumerate(["M0", "M1", "M2"]):
            dm = d[d["mapper"] == m].set_index("rate").loc[rates]
            xs = x + (k - 1) * w
            ax.bar(xs, dm["rho"], width=w, color=MAPPER_COLOR[m], edgecolor="white", linewidth=0.8,
                   label=m, zorder=3)
            ax.errorbar(xs, dm["rho"], yerr=[dm["rho"] - dm["rho_lo"], dm["rho_hi"] - dm["rho"]],
                        fmt="none", ecolor=INK2, elinewidth=0.8, capsize=0, zorder=4)
        ax.axhline(1.0, color=INK, lw=0.9, zorder=5)
        ax.text(-1.02, 1.08, "limit", ha="left", va="bottom", fontsize=FS, color=INK)
        ax.set_xlim(-1.08, len(rates) - 0.55)
        ax.set_yscale("log")
        ax.set_yticks([0.25, 0.5, 1, 2, 4, 8, 16, 32, 64])
        ax.set_yticklabels(["0.25", "0.5", "1", "2", "4", "8", "16", "32", "64"])
        ax.minorticks_off()
        ax.set_xticks(x)
        ax.set_xticklabels([f"{r:g}×" for r in rates])
        ax.set_xlabel("playback speed of the 30 fps clips")
        ax.set_title(title, color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("signing demand\n(s of signing per s of meeting)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, [MAPPER_LABEL[x] for x in lab], loc="outside lower center", ncol=1)
    save(fig, out, "fig2_signing_demand", "Figure 2. Signing demand with a perfect transcript", check)


def fig_bounded(oracle_dir, out, check):
    s = pd.read_csv(os.path.join(oracle_dir, "oracle_summary.csv"))
    s = s[(s["mode"] == "active") & (s["asr_delay"] == 0) & (s["schedule"].str.startswith("bounded"))]
    s = s.assign(maxlag=s["schedule"].str.replace("bounded", "").astype(float))
    fifo = pd.read_csv(os.path.join(oracle_dir, "oracle_summary.csv"))
    fifo = fifo[(fifo["mode"] == "active") & (fifo["schedule"] == "fifo") & (fifo["asr_delay"] == 0) & (fifo["rate"] == 1.0)]
    ceil = fifo.set_index("mapper")["lex_per_min"]
    fig, axes = plt.subplots(1, 2, figsize=fsize(PX_FULL, 78), sharey=True, layout="constrained")
    for ax, rate in zip(axes, [1.0, 1.5]):
        d = s[s["rate"] == rate]
        for m in ("M2", "M0"):
            ax.axhline(ceil[m], color=AXIS, lw=0.9, zorder=1)
        for m in ["M0", "M1", "M2"]:
            dm = d[d["mapper"] == m].sort_values("maxlag")
            ax.fill_between(dm["maxlag"], dm["delivered_lex_per_min_lo"], dm["delivered_lex_per_min_hi"],
                            color=blend(MAPPER_COLOR[m], 0.14), lw=0, zorder=2)
        for m in ["M0", "M1", "M2"]:
            dm = d[d["mapper"] == m].sort_values("maxlag")
            ax.plot(dm["maxlag"], dm["delivered_lex_per_min"], color=MAPPER_COLOR[m], lw=1.6, marker="o",
                    ms=4.5, mec="white", mew=1.0, zorder=3, label=m)
            last = dm.iloc[-1]
            ax.text(last["maxlag"] + 0.14, last["delivered_lex_per_min"], m, va="center", fontsize=FS, color=INK)
        ax.set_xticks([2, 4, 6])
        ax.set_xticklabels(["2 s", "4 s", "6 s"])
        ax.set_xlim(1.7, 6.75)
        ax.set_xlabel("lag limit (maximum sign delay allowed)")
        ax.set_title(f"({'a' if rate == 1.0 else 'b'}) playback speed {rate:g}×", color=INK)
    axes[0].set_ylabel("dictionary signs delivered\nper minute")
    axes[0].set_ylim(0, 62)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, [MAPPER_LABEL[x] for x in lab], loc="outside lower center", ncol=1)
    fig.canvas.draw()       # layout fixed: put each label of a grey line where it clears the data
    renderer = fig.canvas.get_renderer()
    for ax, rate in zip(axes, [1.0, 1.5]):
        d = s[s["rate"] == rate]
        bands = [d[d["mapper"] == m].sort_values("maxlag") for m in ["M0", "M1", "M2"]]
        for m, lab in (("M2", "all dictionary signs, M1 and M2"), ("M0", "all dictionary signs, M0")):
            for va, dy in (("top", -0.9), ("bottom", 0.9)):
                t = ax.text(1.78, ceil[m] + dy, lab, fontsize=FS, color=INK2, va=va, ha="left", zorder=5)
                bb = t.get_window_extent(renderer).transformed(ax.transData.inverted())
                xs = np.linspace(bb.x0, bb.x1, 80)
                clash = False
                for b in bands:     # confidence band, and the line with its markers
                    lo = np.interp(xs, b["maxlag"], b["delivered_lex_per_min_lo"], left=np.nan, right=np.nan)
                    hi = np.interp(xs, b["maxlag"], b["delivered_lex_per_min_hi"], left=np.nan, right=np.nan)
                    mid = np.interp(xs, b["maxlag"], b["delivered_lex_per_min"], left=np.nan, right=np.nan)
                    ok = ~np.isnan(lo)
                    clash |= bool(np.any(ok & (hi + 0.3 > bb.y0) & (lo - 0.3 < bb.y1)))
                    clash |= bool(np.any(ok & (mid + 1.5 > bb.y0) & (mid - 1.5 < bb.y1)))
                if not clash:
                    break
                t.remove()
            else:
                raise RuntimeError(f"no free place for the label '{lab}'")
    save(fig, out, "fig3_bounded_lag_delivery", "Figure 3. Dictionary signs delivered with a lag limit", check)


# ------------------------------------------------------------------------------ streaming figures --
O_COLOR = "#898781"   # reference row (perfect transcript): neutral, never a series hue
SHORT = {"P1-h1.0-c0.25": "P1", "P1v-h1.0-c0.25-vad": "P1v", "P2-sil500": "P2", "P3-la-1.0": "P3",
         "P4-aa-1.0": "P4", "O-oracle": "O"}
LONG = {"P1": "P1 fixed 1 s chunks (deployed)", "P1v": "P1v = P1 + VAD gate", "P2": "P2 pause-based utterances",
        "P3": "P3 LocalAgreement-2", "P4": "P4 AlignAtt", "O": "O perfect transcript"}


def _select(df):
    """The streaming experiment of the paper: the most frequent model and simulation mode."""
    model = df["model"].mode().iloc[0]
    aware = bool(df["comp_aware"].mode().iloc[0])
    return df[(df["model"] == model) & (df["comp_aware"] == aware)], model, aware


def _ordered(df):
    df = df.copy()
    df["_o"] = df["policy"].map({p: i for i, p in enumerate(SHORT)}).fillna(99)
    return df.sort_values("_o").drop(columns="_o")


def fig_tradeoff(stream_dir, out, check):
    t, model, aware = _select(pd.read_csv(os.path.join(stream_dir, "summary_transcript.csv")))
    fig, ax = plt.subplots(figsize=fsize(PX_ONE, 92), layout="constrained")
    for _, r in _ordered(t).iterrows():
        short = SHORT.get(r["policy"], r["policy"])
        fam = short[:2]
        variant = short == "P1v"
        ax.errorbar(r["emit_lag_median"], 100 * r["wer"],
                    xerr=[[r["emit_lag_median"] - r["emit_lag_median_lo"]], [r["emit_lag_median_hi"] - r["emit_lag_median"]]],
                    yerr=[[100 * (r["wer"] - r["wer_lo"])], [100 * (r["wer_hi"] - r["wer"])]],
                    fmt=POLICY_MARKER[fam], color=POLICY_COLOR[fam], ms=6, mec=POLICY_COLOR[fam] if variant else "white",
                    mfc="white" if variant else POLICY_COLOR[fam], mew=1.1,
                    ecolor=POLICY_COLOR[fam], elinewidth=0.8, zorder=3)
        ax.annotate(short, (r["emit_lag_median"], 100 * r["wer"]), xytext=(5, 4 if not variant else -11),
                    textcoords="offset points", fontsize=FS, color=INK)
    ax.set_xlabel("median word delay (s)" if not aware else "median word delay (s), computing time included")
    ax.set_ylabel("word error rate, WER (%)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    names = list(POLICY_NAME)
    handles = [Line2D([], [], ls="", marker=POLICY_MARKER[p], color=POLICY_COLOR[p], ms=5.5, mec="white")
               for p in names]
    labels = [LONG[p] for p in names]
    if (t["policy"] == "P1v-h1.0-c0.25-vad").any():
        handles.insert(1, Line2D([], [], ls="", marker="o", color=POLICY_COLOR["P1"], mfc="white", ms=5.5))
        labels.insert(1, LONG["P1v"])
    fig.legend(handles, labels, loc="outside lower center", ncol=1)
    save(fig, out, "fig5_latency_quality", "Figure 5. Delay and accuracy of the streaming methods", check,
         width_label="one column")


def _color(f_):
    return O_COLOR if f_ == "O" else POLICY_COLOR[f_[:2]]


def _hbar(ax, d, col, scale=1.0, nd=1, unit=""):
    y = np.arange(len(d))
    fams = [SHORT.get(p, p) for p in d["policy"]]
    cols = [_color(f_) for f_ in fams]
    v = scale * d[col].values
    bars = ax.barh(y, v, height=0.58, color=cols, edgecolor="white", linewidth=0.8, zorder=3)
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
        ax.errorbar(v[m], y[m], xerr=[v[m] - lo[m], hi[m] - v[m]], fmt="none", ecolor=INK2, elinewidth=0.8, zorder=4)
        ends = np.where(m, hi, v)
    else:
        ends = v
    for yi, e, val in zip(y, ends, v):
        ax.annotate(f"{val:.{nd}f}{unit}", (e, yi), xytext=(3, 0), textcoords="offset points", va="center",
                    fontsize=FS, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([LONG.get(f_, f_) for f_ in fams])
    if not ax.yaxis_inverted():   # shared y axes: invert only once
        ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(ends) * 1.3 if len(ends) else 1)


def fig_spurious(stream_dir, out, check):
    s, model, aware = _select(pd.read_csv(os.path.join(stream_dir, "summary_sign.csv")))
    d = _ordered(s[(s["schedule"] == "fifo@1x") & (s["mapper"] == "M0")])
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=fsize(PX_FULL, 60), sharey=True, layout="constrained")
    _hbar(axes[0], d, "spurious_fs_per_min", nd=1)
    axes[0].set_xlabel("wrong fingerspelled items per minute")
    axes[0].set_title("(a) wrong fingerspelling", color=INK)
    _hbar(axes[1], d, "rho_hyp", nd=2)
    axes[1].set_xlabel("signing demand at 1×\n(s of signing per s of audio)")
    axes[1].set_title("(b) signing demand", color=INK)
    save(fig, out, "fig6_spurious_fingerspelling", "Figure 6. Wrong fingerspelling and signing demand", check)


def fig_sign_lag(stream_dir, out, check, mapper="M2", schedule="bounded4@1.5x"):
    s, model, aware = _select(pd.read_csv(os.path.join(stream_dir, "summary_sign.csv")))
    d = _ordered(s[(s["mapper"] == mapper) & (s["schedule"] == schedule)])
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=fsize(PX_FULL, 60), sharey=True, layout="constrained")
    _hbar(axes[0], d, "delivered_correct_lex_per_min", nd=1)
    axes[0].set_xlabel("correct dictionary signs\ndelivered per minute")
    axes[0].set_title("(a) correct signs delivered", color=INK)
    _hbar(axes[1], d, "sign_lag_median", nd=2, unit=" s")
    axes[1].set_xlabel("median sign delay\nof delivered signs (s)")
    axes[1].set_title("(b) sign delay", color=INK)
    save(fig, out, "fig7_sign_lag_delivery", "Figure 7. Correct signs delivered and sign delay", check)


# ------------------------------------------------------------------------------------ Figure 4 --
_T2P = TextToPath()
_FP = {False: FontProperties(family=FONT, size=FS), True: FontProperties(family=FONT, size=FS, style="italic")}
_FPB = FontProperties(family=FONT, size=FS, weight="bold")


def _mm(s, italic=False, fp=None):
    """Printed width of a text in mm."""
    return _T2P.get_text_width_height_descent(s, fp or _FP[italic], ismath=False)[0] * 25.4 / 72


_SPACE = _mm("a b") - _mm("ab")


def _wrap(words, max_mm):
    """Greedy line breaking of [(text, fingerspelled)] into lines no wider than max_mm."""
    lines, cur, w = [], [], 0.0
    for txt, fsp in words:
        ww = _mm(txt, fsp)
        add = ww if not cur else _SPACE + ww
        if cur and w + add > max_mm:
            lines.append(cur)
            cur, w = [], 0.0
            add = ww
        cur.append((txt, fsp))
        w += add
    if cur:
        lines.append(cur)
    return lines


def _line_mm(line):
    return sum(_mm(t, f) for t, f in line) + _SPACE * (len(line) - 1)


def fig_example(runs_dir, ref_dir, lexicon, out, check, meeting, t0, t1, model="small"):
    """Released words of each method over a short stretch of one excerpt.  Every release is a tick at its
    release time and the words released together are written after it, coloured by what the deployed rule M0
    makes of them (blue: sign, orange italic: fingerspelled).  Releases whose words would overlap are stacked
    from top to bottom in the order of release, so that no connector crosses any text.  The top row shows the
    words as spoken, with a grey bar for the time during which they were spoken."""
    from itertools import groupby

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from s2sbench.io import load_reference, load_run
    from s2sbench.lexicon import Lexicon
    from s2sbench.mapper import make_mapper
    from s2sbench.metrics import prepare_for_mapping

    lx = Lexicon.from_json(lexicon)
    m0 = make_mapper("M0", lx)

    def fingerspelled(w):
        items = m0.map([w], batch_by_emit=False)
        return bool(items) and items[0].kind != "LEX"

    tracks = []
    start = limit = None
    for pid in ["P1-h1.0-c0.25", "P2-sil500", "P3-la-1.0", "P4-aa-1.0"]:
        fs = glob.glob(os.path.join(runs_dir, model, pid + "*", f"{meeting}.json"))
        if not fs:
            return
        run = load_run(fs[0])
        start, limit = run.get("start_seconds", 0.0), run.get("limit_seconds")
        words = [w for w in prepare_for_mapping(run["words"]) if t0 <= w.emit < t1]
        tracks.append((SHORT[pid], [(emit, [(w.text.strip(), fingerspelled(w)) for w in g])
                                    for emit, g in groupby(words, key=lambda w: w.emit)]))
    ref, _ = load_reference(os.path.join(ref_dir, f"{meeting}.json"), limit=limit, start=start or 0.0)
    ref = sorted([w for w in ref if t0 <= w.start and w.end <= t1], key=lambda w: w.start)
    segments = []                       # stretches of speech: words with gaps under 0.5 s
    for w in ref:
        if segments and w.start - segments[-1]["end"] < 0.5:
            segments[-1]["end"] = max(segments[-1]["end"], w.end)
            segments[-1]["words"].append((w.text.strip(), fingerspelled(w)))
        else:
            segments.append({"start": w.start, "end": w.end, "words": [(w.text.strip(), fingerspelled(w))]})

    # geometry in mm: the axes use seconds horizontally and mm vertically
    W = PX_FULL / 300 * 25.4
    left, right, bottom, top = 3.6, 3.0, 19.5, 1.0
    ax_w = W - left - right
    s_per_mm = (t1 - t0) / ax_w
    LH, BASE, PAD, GAP, TICK = 3.6, 1.9, 2.6, 1.0, 1.0   # level spacing, baseline to text, row padding, run gap, tick

    def layout(runs):
        """runs: [(x_start_s, lines)] in time order -> [(x, lines, bottom_level)], number of levels."""
        placed, burst, burst_end = [], [], -1e9

        def flush():
            total = sum(len(r[1]) for r in burst)
            lvl = total
            for x, lines in burst:
                lvl -= len(lines)
                placed.append((x, lines, lvl))
            return total
        n_lev = 0
        for x, lines in runs:
            end = x + (0.8 + max(_line_mm(ln) for ln in lines)) * s_per_mm
            if burst and x < burst_end + GAP * s_per_mm:
                burst.append((x, lines))
                burst_end = max(burst_end, end)
            else:
                if burst:
                    n_lev = max(n_lev, flush())
                burst, burst_end = [(x, lines)], end
        if burst:
            n_lev = max(n_lev, flush())
        return placed, n_lev

    rows = []
    # speech row: one run per stretch of speech, wrapped to the length of its grey bar
    sp_runs = [(sg["start"], _wrap(sg["words"], max(25.0, (sg["end"] - sg["start"]) / s_per_mm))) for sg in segments]
    rows.append(("Speech, words as spoken", layout(sp_runs), segments, False))
    for name, rels in tracks:
        runs = []
        for emit, words in rels:
            room = (t1 - emit) / s_per_mm - 1.2
            runs.append((emit, _wrap(words, min(60.0, room))))
        rows.append((LONG[name], layout(runs), None, True))

    # label of each row at its top left, above the runs unless the top level is free there
    heights, label_lv = [], []
    for label, (placed, n_lev), _, _ in rows:
        lab_end = t0 + (_mm(label, fp=_FPB) + 2.0) * s_per_mm
        top_lv = max(n_lev - 1, 0)
        busy = any(lvl <= top_lv < lvl + len(lines) and x < lab_end for x, lines, lvl in placed)
        lv = top_lv + 1 if (busy or n_lev == 0) else top_lv
        label_lv.append(lv)
        heights.append(BASE + (lv + 1) * LH + PAD)
    ax_h = sum(heights) + 1.0
    H = bottom + ax_h + top
    fig = plt.figure(figsize=fsize(PX_FULL, H))
    H = fig.get_figheight() * 25.4
    ax = fig.add_axes([left / W, bottom / H, ax_w / W, ax_h / H])
    ax.set_xlim(t0, t1)
    ax.set_ylim(0, ax_h)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    ax.set_xticks(np.arange(np.ceil(t0), np.floor(t1) + 1e-9, 1.0))
    ax.set_xticklabels([f"{v:g}" for v in np.arange(np.ceil(t0), np.floor(t1) + 1e-9, 1.0)])
    y = ax_h
    for (label, (placed, n_lev), segs, is_policy), h, lv_lab in zip(rows, heights, label_lv):
        y -= h
        yb = y + 1.4                                   # baseline of the row
        ax.plot([t0, t1], [yb, yb], color=GRID, lw=0.6, zorder=1, solid_capstyle="butt")
        ax.text(t0 + 0.6 * s_per_mm, yb + BASE + lv_lab * LH, label, fontproperties=_FPB, color=INK,
                ha="left", va="baseline", zorder=6)
        if segs:
            for sg in segs:
                ax.add_patch(Rectangle((sg["start"], yb - 0.7), sg["end"] - sg["start"], 1.4, fc=AXIS, ec="none",
                                       zorder=2))
        for x, lines, lvl in placed:
            if is_policy:
                ax.plot([x, x], [yb - TICK, yb + TICK], color=INK, lw=1.1, zorder=4, solid_capstyle="butt")
                if lvl > 0:
                    ax.plot([x, x], [yb + TICK, yb + BASE + lvl * LH - 0.9], color=INK2, lw=0.6, zorder=3,
                            solid_capstyle="butt")
            for i, line in enumerate(lines):
                ly = yb + BASE + (lvl + len(lines) - 1 - i) * LH
                off = 0.8
                for txt, fsp in line:
                    ax.text(x + off * s_per_mm, ly, txt, fontproperties=_FP[fsp], color=FS_TXT if fsp else SIGN_TXT,
                            ha="left", va="baseline", zorder=5)
                    off += _mm(txt, fsp) + _SPACE
    ax.set_xlabel(f"time (s) within the excerpt of AMI {meeting}, which starts {start:.0f} s into the meeting")
    handles = [Line2D([], [], ls="", marker="s", ms=6, color=SIGN_TXT),
               Line2D([], [], ls="", marker="s", ms=6, color=FS_TXT),
               Line2D([], [], ls="", marker="|", ms=7, mew=1.1, color=INK),
               Patch(fc=AXIS, ec="none")]
    fig.legend(handles, ["word with a sign in the dictionary (M0)", "word fingerspelled by M0 (italic)",
                         "release, followed by the words released", "time during which the words were spoken"],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.004), handlelength=1.2)
    save(fig, out, "fig4_example_commits", "Figure 4. Words released by the four streaming methods", check)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", default="results/oracle")
    ap.add_argument("--streaming", default="results/streaming")
    ap.add_argument("--runs", default="runs_cpu")
    ap.add_argument("--ref", default="data/ami/ref")
    ap.add_argument("--lexicon", default="data/lexicon.json")
    ap.add_argument("--example", nargs=3, default=None, metavar=("MEETING", "T0", "T1"),
                    help="excerpt and time window (s) for the example figure, e.g. ES2004a 35 46.5")
    ap.add_argument("--out", default="manuscript/figures_v4")
    a = ap.parse_args()
    style()
    check = []
    fig_pipeline(a.out, check)
    if os.path.exists(os.path.join(a.oracle, "oracle_summary.csv")):
        fig_demand(a.oracle, a.out, check)
        fig_bounded(a.oracle, a.out, check)
    if a.example:
        fig_example(a.runs, a.ref, a.lexicon, a.out, check, a.example[0], float(a.example[1]), float(a.example[2]))
    if os.path.exists(os.path.join(a.streaming, "summary_transcript.csv")):
        fig_tradeoff(a.streaming, a.out, check)
        fig_spurious(a.streaming, a.out, check)
        fig_sign_lag(a.streaming, a.out, check)
    write_check(a.out, sorted(check, key=lambda c: c["name"]))


if __name__ == "__main__":
    main()
