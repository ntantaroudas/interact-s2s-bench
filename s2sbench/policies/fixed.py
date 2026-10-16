"""P1 -- fixed-length chunking (the INTERACT baseline).

Audio is cut into windows of ``hop`` seconds.  With ``context > 0`` each
window is extended by ``context`` seconds on both sides (so consecutive windows
overlap by ``2*context``) and only words whose midpoint falls inside the
window's core ``[k*hop, (k+1)*hop)`` are kept -- a timestamp-based de-duplication
of the overlap.  With ``context = 0`` windows are disjoint and every word is
kept (naive concatenation).  Window ``k`` can be transcribed once audio up to
``(k+1)*hop + context`` has arrived.  Each window is transcribed independently
(no prompt), as in the deployed system.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..types import Word
from .base import GrowBuffer, OnlineProcessor, clean_words


class FixedChunkProcessor(OnlineProcessor):
    def __init__(self, asr, hop: float = 1.0, context: float = 0.25):
        super().__init__()
        if hop <= 0 or context < 0:
            raise ValueError("hop must be > 0 and context >= 0")
        self.asr = asr
        self.hop = float(hop)
        self.context = float(context)
        self.name = f"P1-fixed(h={hop:g},c={context:g})"
        self.reset()

    def reset(self) -> None:
        self.buf = GrowBuffer()
        self.k = 0
        self.boundaries = []
        self.asr_calls = 0
        self.asr_audio_sec = 0.0

    def insert_audio_chunk(self, audio: np.ndarray) -> None:
        self.buf.append(audio)

    def _ready(self) -> bool:
        return (self.k + 1) * self.hop + self.context <= self.buf.end_time + 1e-9

    def has_pending(self) -> bool:
        return self._ready()

    def _run_window(self, final: bool = False) -> List[Word]:
        k, h, c = self.k, self.hop, self.context
        a = max(0.0, k * h - c)
        b = min(self.buf.end_time, (k + 1) * h + c)
        audio = self.buf.get(a, b)
        words = self.asr.transcribe(audio, offset=a) if len(audio) else []
        self.asr_calls += 1
        self.asr_audio_sec += len(audio) / 16000
        core_lo = k * h
        core_hi = float("inf") if final else (k + 1) * h
        if c > 0:
            words = [w for w in words if core_lo <= w.mid < core_hi]
        self.boundaries.append((k + 1) * h)
        self.k += 1
        self.buf.drop_before(max(0.0, self.k * h - c))
        return clean_words(words)

    def process_iter(self) -> List[Word]:
        if not self._ready():
            return []
        return self._run_window()

    def finish(self) -> List[Word]:
        out: List[Word] = []
        while self._ready():
            out.extend(self._run_window())
        # trailing partial window(s)
        while self.k * self.hop < self.buf.end_time - 1e-6:
            last = (self.k + 1) * self.hop >= self.buf.end_time - 1e-6
            out.extend(self._run_window(final=last))
        return out

    def warmup(self, audio: np.ndarray) -> None:
        self.asr.warmup(audio)
