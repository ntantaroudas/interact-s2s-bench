"""AMI Meeting Corpus helpers: official partitions and NXT word annotations.

The AMI corpus and its annotations are released under CC BY 4.0
(https://groups.inf.ed.ac.uk/ami/corpus/license.shtml).  We use the
*full-corpus-ASR* partition published at
https://groups.inf.ed.ac.uk/ami/corpus/datasets.shtml and the manual word
alignments in ``ami_public_manual_1.6.2`` (``words/<meeting>.<spk>.words.xml``).
"""
from __future__ import annotations

import glob
import html
import os
import re
import xml.etree.ElementTree as ET
from typing import Dict, List

from .types import Word

# Full-corpus-ASR partition (AMI website, "Training, test and evaluation sets").
# Eval (SC): ES2004, IS1009, TS3003 (scenario) + EN2002 (non-scenario).
# Dev  (SB): ES2011, IS1008, TS3004 + IB4001-IB4004, IB4010, IB4011.
_SCEN = "abcd"
EVAL_MEETINGS: List[str] = (
    [f"ES2004{x}" for x in _SCEN]
    + [f"IS1009{x}" for x in _SCEN]
    + [f"TS3003{x}" for x in _SCEN]
    + [f"EN2002{x}" for x in _SCEN]
)
DEV_MEETINGS: List[str] = (
    [f"ES2011{x}" for x in _SCEN]
    + [f"IS1008{x}" for x in _SCEN]
    + [f"TS3004{x}" for x in _SCEN]
    + ["IB4001", "IB4002", "IB4003", "IB4004", "IB4010", "IB4011"]
)
PARTITIONS: Dict[str, List[str]] = {"eval": EVAL_MEETINGS, "dev": DEV_MEETINGS}

AUDIO_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Mix-Headset.wav"
)
ANNOTATIONS_URL = (
    "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
)

# Non-word elements that may appear in the word streams.
_NONWORD_TAGS = {"vocalsound", "nonvocalsound", "disfmarker", "gap", "comment", "pause", "sil"}


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _id_index(nite_id: str) -> int:
    """'ES2004a.A.words12' -> 12."""
    tail = nite_id.rsplit("words", 1)[-1]
    return int(tail) if tail.isdigit() else -1


def parse_words_file(path: str, speaker: str) -> tuple[List[Word], List[tuple]]:
    """Parse one ``*.words.xml`` file.

    Returns (words, nonword_events).  Punctuation tokens (``punc="true"``) are
    dropped.  Truncated words (``trunc="true"``) are kept but flagged in
    ``meta['trunc']`` so callers can decide whether to score them.  Each word
    keeps its NXT index in ``meta['idx']`` (used to attach speaker segments).
    """
    words: List[Word] = []
    events: List[tuple] = []
    tree = ET.parse(path)
    nid = "{http://nite.sourceforge.net/}id"
    for el in tree.getroot():
        tag = _local(el.tag)
        st, et = el.get("starttime"), el.get("endtime")
        if tag == "w":
            if el.get("punc") == "true":
                continue
            if st is None or et is None:
                # a handful of tokens lack times; they cannot be used for timing
                continue
            text = html.unescape((el.text or "").strip())
            if not text:
                continue
            meta = {"idx": _id_index(el.get(nid, ""))}
            if el.get("trunc") == "true":
                meta["trunc"] = True
            words.append(Word(text=text, start=float(st), end=float(et), speaker=speaker, meta=meta))
        elif tag in _NONWORD_TAGS and st is not None and et is not None:
            events.append((tag, float(st), float(et), el.get("type")))
    return words, events


_RANGE = re.compile(r"id\(([^)]+)\)(?:\.\.id\(([^)]+)\))?")


def parse_segments_file(path: str) -> List[tuple]:
    """Speaker segments: list of (seg_index, first_word_idx, last_word_idx)."""
    out = []
    tree = ET.parse(path)
    for k, seg in enumerate(tree.getroot()):
        if _local(seg.tag) != "segment":
            continue
        for ch in seg:
            href = ch.get("href", "")
            m = _RANGE.search(href)
            if m:
                a = _id_index(m.group(1))
                b = _id_index(m.group(2)) if m.group(2) else a
                out.append((k, a, b))
    return out


def load_meeting_words(manual_dir: str, meeting: str, keep_truncated: bool = False,
                       order: str = "segment") -> List[Word]:
    """All speakers' words of a meeting.

    ``manual_dir`` is the unpacked ``ami_public_manual_1.6.2`` directory.
    ``order="segment"`` (default) orders words by the start time of their AMI
    speaker segment and then by word start, i.e. overlapping utterances are
    concatenated whole (as a single-stream ASR transcript would be) instead of
    being interleaved word by word; ``order="word"`` sorts by word start time.
    Each word gets ``meta['seg']`` = (speaker, segment index), ``meta['seg_start']``
    and ``meta['overlap']=True`` if it overlaps a word of another speaker.
    """
    files = sorted(glob.glob(os.path.join(manual_dir, "words", f"{meeting}.*.words.xml")))
    if not files:
        raise FileNotFoundError(f"No word files for {meeting} under {manual_dir}/words")
    allw: List[Word] = []
    for f in files:
        spk = os.path.basename(f).split(".")[1]
        w, _ = parse_words_file(f, spk)
        seg_path = os.path.join(manual_dir, "segments", f"{meeting}.{spk}.segments.xml")
        idx2seg = {}
        if os.path.exists(seg_path):
            for k, a, b in parse_segments_file(seg_path):
                for i in range(a, b + 1):
                    idx2seg[i] = k
        # words outside any segment get their own pseudo-segment
        for x in w:
            k = idx2seg.get(x.meta.get("idx", -1))
            x.meta["seg"] = f"{spk}:{k if k is not None else 'w' + str(x.meta.get('idx'))}"
        allw.extend(w)
    if not keep_truncated:
        allw = [w for w in allw if not w.meta.get("trunc")]
    seg_start: Dict[str, float] = {}
    for x in allw:
        seg_start[x.meta["seg"]] = min(seg_start.get(x.meta["seg"], x.start), x.start)
    for x in allw:
        x.meta["seg_start"] = round(seg_start[x.meta["seg"]], 4)
    allw.sort(key=lambda w: (w.start, w.end, w.speaker or ""))
    _flag_overlaps(allw)
    if order == "segment":
        allw.sort(key=lambda w: (w.meta["seg_start"], w.meta["seg"], w.start, w.end))
    elif order != "word":
        raise ValueError("order must be 'segment' or 'word'")
    for x in allw:
        x.meta.pop("idx", None)
    return allw


def reference_segments(words: List[Word]) -> List[dict]:
    """Speaker segments (for MeetEval): speaker, start, end, text."""
    segs: Dict[str, dict] = {}
    for w in words:
        k = w.meta["seg"]
        d = segs.setdefault(k, {"speaker": w.speaker, "start": w.start, "end": w.end, "words": []})
        d["start"] = min(d["start"], w.start)
        d["end"] = max(d["end"], w.end)
        d["words"].append((w.start, w.text))
    out = []
    for k, d in segs.items():
        d["words"].sort()
        out.append({"speaker": d["speaker"], "start": d["start"], "end": d["end"],
                    "text": " ".join(t for _, t in d["words"])})
    out.sort(key=lambda x: (x["start"], x["speaker"]))
    return out


def _flag_overlaps(words: List[Word]) -> None:
    """Mark words that overlap with any word of a different speaker."""
    # sweep: words sorted by start; keep active intervals per speaker
    active: List[Word] = []
    for w in words:
        active = [a for a in active if a.end > w.start]
        for a in active:
            if a.speaker != w.speaker and a.end > w.start and w.end > a.start:
                a.meta["overlap"] = True
                w.meta["overlap"] = True
        active.append(w)


def speech_intervals(words: List[Word], merge_gap: float = 0.0) -> List[tuple]:
    """Union of reference word intervals (any speaker), optionally merging gaps."""
    iv = sorted((w.start, w.end) for w in words)
    out: List[list] = []
    for s, e in iv:
        if out and s <= out[-1][1] + merge_gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]
