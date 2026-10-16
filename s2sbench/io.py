"""Loading reference and run files."""
from __future__ import annotations

import json
from typing import List, Optional, Tuple

from .normalize import FILLERS, strip_for_lookup
from .types import Word


def load_reference(path: str, limit: Optional[float] = None, drop_fillers: bool = True,
                   start: float = 0.0) -> Tuple[List[Word], List[dict]]:
    """Reference words (segment order) and speaker segments.

    ``limit`` keeps only material whose midpoint lies before ``limit`` seconds
    (used when a run processed only the first part of the audio).
    ``start`` (> 0) selects an excerpt: words whose midpoint lies in
    [start, start + limit) are kept and all times are shifted by -start, so they
    match a run on the excerpt audio (which starts at 0).  Speaker segments are
    then rebuilt from the kept words, so a segment cut by the excerpt edge keeps
    only its words inside the excerpt.
    ``drop_fillers`` removes hesitations/back-channels (uh, um, mm-hmm, ...),
    which carry no lexical content and which Whisper normally omits.
    """
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    words = [Word.from_dict(x) for x in d["words"]]
    segs = d.get("segments", [])
    if start and start > 0:
        hi = start + limit if limit is not None else float("inf")
        words = [w for w in words if start <= w.mid < hi]
        for w in words:
            w.start -= start
            w.end -= start
        segs = _segments_from_words(words)
    elif limit is not None:
        words = [w for w in words if w.mid < limit]
        segs = [s for s in segs if 0.5 * (s["start"] + s["end"]) < limit]
    if drop_fillers:
        words = [w for w in words if strip_for_lookup(w.text) not in FILLERS]
    return words, segs


def _segments_from_words(words: List[Word]) -> List[dict]:
    """Group words by their AMI segment id (``meta['seg']``) into speaker segments."""
    groups: dict = {}
    order: List[str] = []
    for w in words:
        sid = w.meta.get("seg") or f"{w.speaker}:{w.start:.2f}"
        if sid not in groups:
            groups[sid] = []
            order.append(sid)
        groups[sid].append(w)
    out = []
    for sid in order:
        g = groups[sid]
        out.append({"speaker": g[0].speaker, "start": min(w.start for w in g), "end": max(w.end for w in g),
                    "text": " ".join(w.text for w in g)})
    return out


def load_run(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    d["words"] = [Word.from_dict(x) for x in d["words"]]
    return d
