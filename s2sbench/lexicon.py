"""The INTERACT International Sign (IS) lexicon as used by the avatar.

The deployed system stores one MediaPipe landmark sequence per lexicon entry
in Redis under ``filename.split('.')[0].lower()`` (see ``populate-redis.py`` in
https://github.com/ntantaroudas/isl-extractions-main).  We reproduce that key
rule exactly, including its side effects (case-duplicates collapse onto one
key; entries such as ``"I (not letter)"`` can never be matched by a single
lower-cased word).

Clip timing.  A landmark file holds every video frame, including frames
before/after the signer is visible.  ``duration(mode="full")`` plays all
frames; ``duration(mode="active")`` plays only the span between the first and
last frame in which MediaPipe detected at least one hand.
"""
from __future__ import annotations

import json
import os
import string
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional

LETTERS = tuple(string.ascii_lowercase)


@dataclass
class Entry:
    key: str
    file: str
    frames: int
    active_first: Optional[int]
    active_last: Optional[int]

    @property
    def active_frames(self) -> int:
        if self.active_first is None or self.active_last is None:
            return self.frames  # no hand detected anywhere: fall back to the full clip
        return self.active_last - self.active_first + 1


def scan_landmarks(landmarks_dir: str) -> List[Entry]:
    """Read every ``*.json`` landmark file and measure total/active frames."""
    entries: List[Entry] = []
    for fn in sorted(os.listdir(landmarks_dir)):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(landmarks_dir, fn), encoding="utf-8") as f:
            frames = json.load(f)
        has = [
            (fr.get("left_hand_landmarks") is not None) or (fr.get("right_hand_landmarks") is not None)
            for fr in frames
        ]
        idx = [i for i, h in enumerate(has) if h]
        key = fn.split(".")[0].lower()  # identical to populate-redis.py
        entries.append(
            Entry(
                key=key,
                file=fn,
                frames=len(frames),
                active_first=idx[0] if idx else None,
                active_last=idx[-1] if idx else None,
            )
        )
    return entries


class Lexicon:
    """Key -> clip lookup with clip durations."""

    def __init__(self, entries: Iterable[Entry], fps: float = 30.0, mode: str = "active"):
        if mode not in ("active", "full"):
            raise ValueError("mode must be 'active' or 'full'")
        self.fps = float(fps)
        self.mode = mode
        self.entries: List[Entry] = list(entries)
        self.by_key: Dict[str, Entry] = {}
        self.collisions: Dict[str, List[str]] = {}
        # Redis SET semantics: the last write wins.  populate-redis.py iterates
        # os.listdir() (unordered); we fix the order to sorted file names so the
        # benchmark is deterministic, and report the colliding keys.
        for e in sorted(self.entries, key=lambda x: x.file):
            if e.key in self.by_key:
                self.collisions.setdefault(e.key, [self.by_key[e.key].file]).append(e.file)
            self.by_key[e.key] = e
        self.max_phrase_len = max(len(k.split()) for k in self.by_key)

    # ------------------------------------------------------------------ I/O
    @classmethod
    def from_json(cls, path: str, fps: float = 30.0, mode: str = "active") -> "Lexicon":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls([Entry(**e) for e in data["entries"]], fps=fps, mode=mode)

    @classmethod
    def from_landmarks(cls, landmarks_dir: str, fps: float = 30.0, mode: str = "active") -> "Lexicon":
        return cls(scan_landmarks(landmarks_dir), fps=fps, mode=mode)

    def to_json(self, path: str, source: str = "") -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"source": source, "n_files": len(self.entries), "entries": [asdict(e) for e in self.entries]},
                f,
                indent=1,
            )

    # -------------------------------------------------------------- queries
    def __contains__(self, key: str) -> bool:
        return key in self.by_key

    def duration(self, key: str) -> float:
        e = self.by_key[key]
        n = e.active_frames if self.mode == "active" else e.frames
        return n / self.fps

    def letter_key(self, ch: str) -> Optional[str]:
        """Clip key used to fingerspell one character (letters and digits 1-9).

        The lexicon has number signs for 1-10 but no 0; we spell '0' with the
        letter O, which is the conventional fallback in one-handed manual
        alphabets.  Other characters are skipped.
        """
        ch = ch.lower()
        if ch in LETTERS and ch in self.by_key:
            return ch
        if ch.isdigit():
            if ch == "0":
                return "o" if "o" in self.by_key else None
            return ch if ch in self.by_key else None
        return None

    def stats(self) -> dict:
        keys = list(self.by_key)
        return {
            "files": len(self.entries),
            "unique_keys": len(keys),
            "collisions": self.collisions,
            "letters": sum(1 for k in keys if k in LETTERS),
            "numbers": sum(1 for k in keys if k.isdigit()),
            "multiword": sum(1 for k in keys if " " in k),
            "unreachable_single_token": sorted(k for k in keys if any(c in k for c in "()")),
        }
