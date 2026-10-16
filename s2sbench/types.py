"""Core data types shared by all modules."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Word:
    """A word with audio timestamps.

    For reference words, ``start``/``end`` come from the AMI manual word
    alignment.  For hypothesis words they are the ASR's own word timestamps
    (stream time, seconds from the start of the meeting audio).  ``emit`` is
    the (virtual) wall-clock time at which a streaming policy committed the
    word; it is ``None`` for reference words.
    """

    text: str
    start: float
    end: float
    emit: Optional[float] = None
    speaker: Optional[str] = None
    meta: dict = field(default_factory=dict)

    @property
    def mid(self) -> float:
        return 0.5 * (self.start + self.end)

    def to_dict(self) -> dict:
        d = {"text": self.text, "start": round(self.start, 4), "end": round(self.end, 4)}
        if self.emit is not None:
            d["emit"] = round(self.emit, 4)
        if self.speaker is not None:
            d["speaker"] = self.speaker
        if self.meta:
            d["meta"] = self.meta
        return d

    @staticmethod
    def from_dict(d: dict) -> "Word":
        return Word(
            text=d["text"],
            start=float(d["start"]),
            end=float(d["end"]),
            emit=None if d.get("emit") is None else float(d["emit"]),
            speaker=d.get("speaker"),
            meta=d.get("meta", {}) or {},
        )
