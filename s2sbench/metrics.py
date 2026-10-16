"""Transcript-, sign- and timing-level metrics.

Definitions (all computed per meeting):

WER            jiwer word error rate after Whisper's EnglishTextNormalizer on the
               whole meeting transcript (reference: AMI manual words, all
               speakers, time-ordered; truncated words removed).
Emission lag   for every hypothesis token aligned (equal) to a reference token:
               ``emit - ref_end``; pairs whose ASR timestamp is > 3 s from the
               reference word are treated as accidental matches and skipped.
FRAG           fragments per 1000 reference tokens: a substituted hypothesis
               token that is a proper prefix/suffix (>= 1 char) of the reference
               token it replaces, or two adjacent hypothesis tokens whose
               concatenation equals a reference token.
DUP            inserted hypothesis tokens equal to an adjacent hypothesis
               token, per 1000 reference tokens (overlap/merge duplicates).
NONSPEECH      hypothesis tokens whose ASR timestamp midpoint is more than
               1 s away from any reference word (any speaker), per hour --
               a proxy for hallucinated output in silence/noise.
SER            sign error rate: Levenshtein distance between the sign-token
               sequence produced from the hypothesis and the one produced from
               the reference by the *same* mapper, divided by the number of
               reference sign tokens.
Spurious FS    fingerspelled items in the hypothesis sign sequence that are
               insertions or substitutions in the sign alignment, per minute.
Sign lag       ``onset - ref_end`` for hypothesis sign items aligned (equal) to a
               reference sign item and actually played; |ASR ts - ref ts| <= 5 s.
Delivered      correctly signed and played reference sign items / all reference
               sign items (sign recall under a scheduling policy).
"""
from __future__ import annotations

import bisect
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .avatar import Played, schedule, demand_ratio
from .mapper import SignItem
from .normalize import expand_words, wer_normalize
from .types import Word


# ----------------------------------------------------------------- helpers --
def join_continuations(words: Sequence[Word]) -> List[Word]:
    """Re-join word pieces flagged ``meta['cont']`` (SimulStreaming) onto the
    previous word for transcript-level scoring."""
    out: List[Word] = []
    for w in words:
        if w.meta.get("cont") and out:
            p = out[-1]
            out[-1] = Word(p.text + w.text, p.start, max(p.end, w.end), emit=w.emit, speaker=p.speaker,
                           meta={k: v for k, v in p.meta.items() if k != "cont"})
        else:
            out.append(Word(w.text, w.start, w.end, emit=w.emit, speaker=w.speaker,
                            meta={k: v for k, v in w.meta.items() if k != "cont"}))
    return out


def drop_annotations(words: Sequence[Word], max_words: int = 4) -> List[Word]:
    """Remove bracketed sound descriptions that Whisper sometimes emits instead of words, e.g.
    "[BLANK_AUDIO]", "(laughing)", "[INTERPOSING VOICES]", "(keyboard clacking)".  Whisper's English
    normaliser removes the same spans before WER scoring.

    A span starts at a word beginning with "[" or "(" and ends at the word carrying the closing bracket.
    Word-piece outputs may lose the closing bracket (a bracket-only piece is not a word), so a span also
    ends at the first following word that is not part of the same commit, or that does not look like a
    sound description ("[...]": upper case; "(...)": lower case).  Spans are at most ``max_words`` long."""
    out: List[Word] = []
    i, n = 0, len(words)
    while i < n:
        t = words[i].text.strip()
        if t[:1] not in "[(":
            out.append(words[i])
            i += 1
            continue
        closing = "]" if t[0] == "[" else ")"
        j = i + 1
        if not t.rstrip(".,!?;:").endswith(closing):
            while j < n and j - i < max_words:
                u = words[j].text.strip()
                core = u.strip("[]().,!?;:_'-")
                if u.rstrip(".,!?;:").endswith(closing):
                    j += 1
                    break
                same_commit = words[j].emit == words[i].emit
                looks = core.isupper() if closing == "]" else (core.islower() and core.isalpha())
                if not (same_commit and looks):
                    break
                j += 1
        i = j
    return out


def prepare_for_mapping(words: Sequence[Word]) -> List[Word]:
    """What a text-to-sign front end receives: whole words (word pieces re-joined; a continuation piece
    delays its word to the piece's emission time) without bracketed sound descriptions."""
    return drop_annotations(join_continuations(words))


def _align(ref_tokens: List[str], hyp_tokens: List[str]):
    import jiwer

    if not ref_tokens:
        return None
    out = jiwer.process_words(" ".join(ref_tokens), " ".join(hyp_tokens) if hyp_tokens else "")
    return out


def _pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def _summ(x: List[float], prefix: str) -> Dict[str, float]:
    x = list(x)
    return {
        f"{prefix}_n": len(x),
        f"{prefix}_median": float(np.median(x)) if x else float("nan"),
        f"{prefix}_mean": float(np.mean(x)) if x else float("nan"),
        f"{prefix}_p90": _pct(x, 90),
        f"{prefix}_p95": _pct(x, 95),
    }


class SpeechIndex:
    """Fast 'distance to nearest reference speech' queries."""

    def __init__(self, ref: Sequence[Word]):
        iv = sorted((w.start, w.end) for w in ref)
        merged: List[List[float]] = []
        for s, e in iv:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        self.starts = [m[0] for m in merged]
        self.iv = merged

    def distance(self, t: float) -> float:
        i = bisect.bisect_right(self.starts, t) - 1
        d = float("inf")
        for j in (i, i + 1):
            if 0 <= j < len(self.iv):
                s, e = self.iv[j]
                d = min(d, 0.0 if s <= t <= e else min(abs(t - s), abs(t - e)))
        return d


# ------------------------------------------------------- meeting-level WER --
def tcorc_wer(ref_segments: Sequence[dict], hyp: Sequence[Word], collar: float = 5.0,
              hyp_gap: float = 0.5) -> Dict[str, float]:
    """Time-constrained optimal-reference-combination WER (MeetEval; von Neumann
    et al.).  Reference: AMI speaker segments; hypothesis: one output stream,
    grouped into pseudo-segments at ASR-timestamp gaps > ``hyp_gap`` s.  Both
    sides use Whisper's EnglishTextNormalizer at segment level."""
    try:
        import meeteval
        from meeteval.io import SegLST
    except Exception:  # pragma: no cover - optional dependency
        return {"tcorc_wer": float("nan")}
    rs = []
    for s in ref_segments:
        t = wer_normalize(s["text"])
        if t.strip():
            rs.append({"session_id": "m", "speaker": s["speaker"], "start_time": float(s["start"]),
                       "end_time": float(s["end"]), "words": t})
    hs: List[dict] = []
    cur: List[Word] = []
    for w in sorted(join_continuations(hyp), key=lambda x: x.start):
        if cur and w.start - cur[-1].end > hyp_gap:
            hs.append(cur)
            cur = []
        cur.append(w)
    if cur:
        hs.append(cur)
    hseg = []
    for g in hs:
        t = wer_normalize(" ".join(w.text for w in g))
        if t.strip():
            hseg.append({"session_id": "m", "speaker": "X", "start_time": float(g[0].start),
                         "end_time": float(max(g[-1].end, g[0].start + 0.01)), "words": t})
    if not rs:
        return {"tcorc_wer": float("nan")}
    if not hseg:
        return {"tcorc_wer": 1.0}
    r = meeteval.wer.tcorcwer(SegLST(rs), SegLST(hseg), collar=collar)
    e = list(r.values())[0]
    return {"tcorc_wer": float(e.error_rate), "tcorc_ins": e.insertions, "tcorc_del": e.deletions,
            "tcorc_sub": e.substitutions, "tcorc_nref": e.length}


# --------------------------------------------------------- transcript level --
def transcript_metrics(ref: Sequence[Word], hyp_raw: Sequence[Word], duration: float,
                       match_window: float = 3.0, nonspeech_gap: float = 1.0,
                       ref_segments: Optional[Sequence[dict]] = None) -> Dict[str, float]:
    hyp = join_continuations(hyp_raw)
    res: Dict[str, float] = {}
    if ref_segments:
        res.update(tcorc_wer(ref_segments, hyp_raw))
    # headline WER (Whisper normaliser on whole transcripts)
    import jiwer

    r_txt = wer_normalize(" ".join(w.text for w in ref))
    h_txt = wer_normalize(" ".join(w.text for w in hyp))
    o = jiwer.process_words(r_txt, h_txt if h_txt.strip() else "")
    n_ref = o.hits + o.substitutions + o.deletions
    res.update({"wer": o.wer, "wer_sub": o.substitutions, "wer_del": o.deletions, "wer_ins": o.insertions,
                "wer_nref": n_ref, "hyp_words": len(hyp)})

    # token-level alignment with timing
    rt, rsrc = expand_words(ref)
    ht, hsrc = expand_words(hyp)
    res["tok_nref"], res["tok_nhyp"] = len(rt), len(ht)
    al = _align(rt, ht)
    lat: List[float] = []
    frag = dup = 0
    if al is not None:
        for ch in al.alignments[0]:
            if ch.type == "equal":
                for k in range(ch.ref_end_idx - ch.ref_start_idx):
                    rw, hw = ref[rsrc[ch.ref_start_idx + k]], hyp[hsrc[ch.hyp_start_idx + k]]
                    if hw.emit is not None and abs(hw.mid - rw.mid) <= match_window:
                        lat.append(hw.emit - rw.end)
            elif ch.type == "substitute":
                for k in range(ch.ref_end_idx - ch.ref_start_idx):
                    r_tok, h_tok = rt[ch.ref_start_idx + k], ht[ch.hyp_start_idx + k]
                    if h_tok != r_tok and len(r_tok) >= 3 and (r_tok.startswith(h_tok) or r_tok.endswith(h_tok)):
                        frag += 1
                    else:
                        # split: this hyp token + its neighbour form the reference token
                        j = ch.hyp_start_idx + k
                        if j + 1 < len(ht) and ht[j] + ht[j + 1] == r_tok:
                            frag += 2
                        elif j - 1 >= 0 and ht[j - 1] + ht[j] == r_tok:
                            frag += 2
            elif ch.type == "insert":
                for j in range(ch.hyp_start_idx, ch.hyp_end_idx):
                    if (j > 0 and ht[j] == ht[j - 1]) or (j + 1 < len(ht) and ht[j] == ht[j + 1]):
                        dup += 1
    res["frag_per_1k"] = 1000.0 * frag / max(len(rt), 1)
    res["dup_per_1k"] = 1000.0 * dup / max(len(rt), 1)
    sidx = SpeechIndex(ref)
    ns = sum(1 for j in range(len(ht)) if sidx.distance(hyp[hsrc[j]].mid) > nonspeech_gap)
    res["nonspeech_tok_per_h"] = ns / (duration / 3600.0)
    res.update(_summ(lat, "emit_lag"))
    for thr in (1.0, 2.0, 4.0):
        res[f"emit_lag_le_{thr:g}s"] = float(np.mean([x <= thr for x in lat])) if lat else float("nan")
    return res


# --------------------------------------------------------------- sign level --
def map_reference(mapper, ref: Sequence[Word]) -> List[SignItem]:
    items = mapper.map(ref, batch_by_emit=False)
    for it in items:  # zero-latency availability (oracle ASR)
        it.emit = ref[it.src[-1]].end
    return items


def sign_metrics(mapper, ref: Sequence[Word], hyp: Sequence[Word], duration: float,
                 rate: float = 1.0, policy: str = "fifo", max_lag: Optional[float] = None,
                 ref_items: Optional[List[SignItem]] = None, match_window: float = 5.0) -> Dict[str, float]:
    ref_items = ref_items if ref_items is not None else map_reference(mapper, ref)
    # only items the avatar plays (LEX, FS) are scored; caption-only items are not signs
    ref_items = [i for i in ref_items if i.played]
    hyp_items = [i for i in mapper.map(prepare_for_mapping(hyp), batch_by_emit=True) if i.played]
    played = schedule(hyp_items, rate=rate, policy=policy, max_lag=max_lag)
    minutes = duration / 60.0
    rtok = [it.token for it in ref_items]
    htok = [it.token for it in hyp_items]
    res: Dict[str, float] = {"n_ref_items": len(rtok), "n_hyp_items": len(htok)}
    al = _align(rtok, htok)
    lat: List[float] = []
    delivered = delivered_lex = 0
    spur_fs = spur_lex = 0
    if al is not None:
        res.update({"ser": al.wer, "ser_sub": al.substitutions, "ser_del": al.deletions, "ser_ins": al.insertions})
        for ch in al.alignments[0]:
            if ch.type == "equal":
                for k in range(ch.ref_end_idx - ch.ref_start_idx):
                    ri, hj = ref_items[ch.ref_start_idx + k], ch.hyp_start_idx + k
                    p = played[hj]
                    if abs(hyp_items[hj].ts_end - ri.ts_end) <= match_window and p.start is not None:
                        delivered += 1
                        delivered_lex += int(ri.kind == "LEX")
                        lat.append(p.start - ri.ts_end)
            elif ch.type in ("insert", "substitute"):
                for j in range(ch.hyp_start_idx, ch.hyp_end_idx):
                    if hyp_items[j].kind == "FS":
                        spur_fs += 1
                    elif hyp_items[j].kind == "LEX":
                        spur_lex += 1
    res["delivered"] = delivered / max(len(rtok), 1)
    res["delivered_correct_lex_per_min"] = delivered_lex / minutes
    res["played_items_per_min"] = sum(1 for p in played if p.start is not None) / minutes
    res["spurious_fs_per_min"] = spur_fs / minutes
    res["spurious_lex_per_min"] = spur_lex / minutes
    res["fs_items_per_min_hyp"] = sum(1 for i in hyp_items if i.kind == "FS") / minutes
    res["fs_items_per_min_ref"] = sum(1 for i in ref_items if i.kind == "FS") / minutes
    res["letters_per_min_hyp"] = sum(len(i.clips) for i in hyp_items if i.kind == "FS") / minutes
    res["rho_hyp"] = demand_ratio(hyp_items, duration, rate)
    res["rho_ref"] = demand_ratio(ref_items, duration, rate)
    n_play = sum(1 for p in played if p.item.played)
    res["drop_rate"] = sum(1 for p in played if p.dropped) / max(n_play, 1)
    res.update(_summ(lat, "sign_lag"))
    for thr in (2.0, 4.0, 6.0):
        res[f"sign_lag_le_{thr:g}s"] = float(np.mean([x <= thr for x in lat])) if lat else float("nan")
    ends = [p.end for p in played if p.end is not None]
    res["overrun_s"] = (max(ends) - duration) if ends else float("nan")
    return res


# ------------------------------------------------------------ oracle (RQ1) --
def oracle_metrics(mapper, ref: Sequence[Word], duration: float, rate: float = 1.0,
                   policy: str = "fifo", max_lag: Optional[float] = None,
                   asr_delay: float = 0.0) -> Tuple[Dict[str, float], List[Played]]:
    """Perfect, zero-latency transcript (plus a constant ``asr_delay``): isolates
    the mapping and the avatar's articulation throughput."""
    items = map_reference(mapper, ref)
    for it in items:
        it.emit = it.emit + asr_delay
    played = schedule(items, rate=rate, policy=policy, max_lag=max_lag)
    lat = [p.start - p.item.ts_end for p in played if p.start is not None]
    n_play = sum(1 for i in items if i.played)
    minutes = duration / 60.0
    res = {
        "n_items": len(items),
        "delivered_lex_per_min": sum(1 for p in played if p.start is not None and p.item.kind == "LEX") / minutes,
        "delivered_items_per_min": sum(1 for p in played if p.start is not None) / minutes,
        "lex_items": sum(1 for i in items if i.kind == "LEX"),
        "fs_items": sum(1 for i in items if i.kind == "FS"),
        "cap_items": sum(1 for i in items if i.kind == "CAP"),
        "letters": sum(len(i.clips) for i in items if i.kind == "FS"),
        "rho": demand_ratio(items, duration, rate),
        "delivered": sum(1 for p in played if p.start is not None) / max(n_play, 1),
    }
    res.update(_summ(lat, "sign_lag"))
    for thr in (2.0, 4.0, 6.0):
        res[f"sign_lag_le_{thr:g}s"] = float(np.mean([x <= thr for x in lat])) if lat else float("nan")
    ends = [p.end for p in played if p.end is not None]
    res["overrun_s"] = (max(ends) - duration) if ends else float("nan")
    return res, played
