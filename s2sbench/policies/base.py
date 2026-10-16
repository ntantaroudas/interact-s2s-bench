"""Common interface for streaming (online) ASR policies.

Every policy exposes the interface used by Whisper-Streaming and
SimulStreaming (``insert_audio_chunk`` / ``process_iter`` / ``finish``) so that
one virtual-clock driver (``s2sbench.simulate``) can run all of them.
``process_iter`` and ``finish`` return the *newly committed* words, each with
absolute stream timestamps.
"""
from __future__ import annotations

from typing import List

import numpy as np

from ..types import Word

SR = 16000


class GrowBuffer:
    """Append-only float32 audio buffer with O(1) amortised appends and an
    offset, so that policies can drop audio they no longer need."""

    def __init__(self):
        self._a = np.zeros(SR * 60, dtype=np.float32)
        self._n = 0          # valid samples in _a
        self.offset = 0      # absolute sample index of _a[0]

    @property
    def end_sample(self) -> int:
        return self.offset + self._n

    @property
    def end_time(self) -> float:
        return self.end_sample / SR

    def append(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float32)
        need = self._n + len(x)
        if need > len(self._a):
            cap = max(need, 2 * len(self._a))
            b = np.zeros(cap, dtype=np.float32)
            b[: self._n] = self._a[: self._n]
            self._a = b
        self._a[self._n: need] = x
        self._n = need

    def get(self, t0: float, t1: float) -> np.ndarray:
        """Audio between absolute times t0 and t1 (clipped to what is held)."""
        i0 = max(int(round(t0 * SR)) - self.offset, 0)
        i1 = min(int(round(t1 * SR)) - self.offset, self._n)
        if i1 <= i0:
            return np.zeros(0, dtype=np.float32)
        return self._a[i0:i1].copy()

    def drop_before(self, t: float) -> None:
        k = int(round(t * SR)) - self.offset
        if k <= 0:
            return
        k = min(k, self._n)
        self._a[: self._n - k] = self._a[k: self._n]
        self._n -= k
        self.offset += k


class OnlineProcessor:
    """Base class for our own policies (P1, P2)."""

    name = "base"
    tick = 0.04  # driver step in seconds (audio arrives in 40 ms packets)

    def __init__(self):
        self.boundaries: List[float] = []
        self.asr_calls = 0
        self.asr_audio_sec = 0.0

    def insert_audio_chunk(self, audio: np.ndarray) -> None:  # pragma: no cover
        raise NotImplementedError

    def process_iter(self) -> List[Word]:  # pragma: no cover
        raise NotImplementedError

    def has_pending(self) -> bool:
        return False

    def finish(self) -> List[Word]:  # pragma: no cover
        raise NotImplementedError

    def warmup(self, audio: np.ndarray) -> None:
        pass

    def reset(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def stats(self) -> dict:
        return {"asr_calls": self.asr_calls, "asr_audio_sec": round(self.asr_audio_sec, 3),
                "n_boundaries": len(self.boundaries)}


def clean_words(words) -> List[Word]:
    """Drop empty / punctuation-only tokens that some back-ends emit as words."""
    out = []
    for w in words:
        t = w.text.strip()
        if t and any(ch.isalnum() for ch in t):
            w.text = t
            out.append(w)
    return out
