"""Sequential avatar playback model.

The avatar plays one clip at a time.  An item becomes available at its
``emit`` time (when the ASR policy committed its source word(s)); it starts at
``max(emit, end_of_previous_item + gap)`` and lasts ``duration / rate``.

Scheduling policies
-------------------
fifo     play everything in order (unbounded queue) -- the deployed behaviour.
bounded  before starting an item, compute the lag the system can observe,
         ``now - ts_end`` (ASR word end timestamp; stream time equals wall-clock
         time for live audio).  If it exceeds ``max_lag`` the item is dropped
         (left to the caption channel) and the next item is considered.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .mapper import SignItem


@dataclass
class Played:
    item: SignItem
    start: Optional[float]
    end: Optional[float]
    dropped: bool = False

    @property
    def queue_delay(self) -> Optional[float]:
        if self.start is None or self.item.emit is None:
            return None
        return self.start - self.item.emit


def schedule(
    items: Sequence[SignItem],
    rate: float = 1.0,
    gap: float = 0.0,
    policy: str = "fifo",
    max_lag: Optional[float] = None,
) -> List[Played]:
    if rate <= 0:
        raise ValueError("rate must be > 0")
    if policy not in ("fifo", "bounded"):
        raise ValueError("policy must be 'fifo' or 'bounded'")
    if policy == "bounded" and max_lag is None:
        raise ValueError("bounded policy needs max_lag")
    order = sorted(range(len(items)), key=lambda i: (items[i].emit if items[i].emit is not None else items[i].ts_end, i))
    out: List[Played] = [None] * len(items)  # type: ignore
    t_free: Optional[float] = None
    for i in order:
        it = items[i]
        if not it.played:
            out[i] = Played(it, None, None, dropped=False)
            continue
        avail = it.emit if it.emit is not None else it.ts_end
        start = avail if t_free is None else max(avail, t_free + gap)
        if policy == "bounded" and (start - it.ts_end) > max_lag:
            out[i] = Played(it, None, None, dropped=True)
            continue
        end = start + it.duration / rate
        out[i] = Played(it, start, end, dropped=False)
        t_free = end
    return out


def utilization(played: Sequence[Played], audio_duration: float) -> float:
    """Fraction of the meeting during which the avatar is signing (can exceed 1
    for an unbounded queue: the avatar keeps signing after the audio ends)."""
    busy = sum(p.end - p.start for p in played if p.start is not None)
    return busy / audio_duration if audio_duration > 0 else float("nan")


def demand_ratio(items: Sequence[SignItem], audio_duration: float, rate: float = 1.0) -> float:
    """Signing time requested per second of audio (rho).  rho > 1 means the
    queue cannot be stable: the avatar falls further behind as the meeting goes on."""
    return sum(i.duration for i in items if i.played) / rate / audio_duration
