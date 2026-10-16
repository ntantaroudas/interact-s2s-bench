"""Virtual-clock simulation of live streaming.

This reproduces the *computationally aware* real-time simulation of
Whisper-Streaming/SimulStreaming (``main_simulation_from_file``) without
sleeping: audio arrives in real time; whenever the processor is idle it takes
all audio that has arrived (at least one ``tick``), runs, and its outputs are
time-stamped with the moment the computation finishes.  Compute time is the
measured wall-clock time of each call on the actual hardware.  Because the
processor is the only user of the GPU, waiting is unnecessary and a 40-minute
meeting is simulated in (real-time factor x 40) minutes.

With ``comp_aware=False`` the clock does not advance during computation
(the "computationally unaware" lower bound on latency).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .types import Word

SR = 16000


@dataclass
class SimResult:
    words: List[Word]
    duration: float
    compute_total: float
    n_calls: int
    iterations: List[dict] = field(default_factory=list)
    boundaries: List[float] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        return self.compute_total / self.duration if self.duration > 0 else float("nan")


def _cuda_sync() -> Optional[Callable[[], None]]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.synchronize
    except Exception:  # pragma: no cover
        pass
    return None


def simulate(proc, audio: np.ndarray, comp_aware: bool = True, tick: Optional[float] = None,
             log_every_call: bool = False) -> SimResult:
    D = len(audio) / SR
    tick = float(tick if tick is not None else getattr(proc, "tick", 0.04))
    sync = _cuda_sync()
    t = 0.0       # virtual wall-clock
    end = 0.0     # audio delivered so far (s)
    words: List[Word] = []
    iters: List[dict] = []
    total = 0.0
    n = 0

    def record(out: List[Word], c: float, audio_end: float):
        nonlocal total, n
        total += c
        n += 1
        for w in out:
            w.emit = t
        words.extend(out)
        if log_every_call or out or c > 0.005:
            iters.append({"t": round(t, 4), "audio_end": round(audio_end, 4), "c": round(c, 5), "n": len(out)})

    while True:
        t = max(t, end + tick)
        new_end = min(t, D)
        chunk = audio[int(round(end * SR)): int(round(new_end * SR))]
        end = new_end
        c0 = time.perf_counter()
        proc.insert_audio_chunk(chunk)
        out = proc.process_iter()
        if sync:
            sync()
        c = time.perf_counter() - c0
        if comp_aware:
            t += c
        record(out, c, end)
        while getattr(proc, "has_pending", lambda: False)():
            c0 = time.perf_counter()
            out = proc.process_iter()
            if sync:
                sync()
            c = time.perf_counter() - c0
            if comp_aware:
                t += c
            record(out, c, end)
        if end >= D - 1e-9:
            break
    c0 = time.perf_counter()
    out = proc.finish()
    if sync:
        sync()
    c = time.perf_counter() - c0
    if comp_aware:
        t += c
    record(out, c, end)
    st = proc.stats() if hasattr(proc, "stats") else {}
    return SimResult(words=words, duration=D, compute_total=total, n_calls=n, iterations=iters,
                     boundaries=list(getattr(proc, "boundaries", [])), stats=st)
