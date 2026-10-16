import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s2sbench.lexicon import Entry, Lexicon  # noqa: E402


@pytest.fixture
def lexicon():
    """A toy lexicon: every clip is 30 active frames (1.0 s at 30 fps), letters 15 frames."""
    ents = []
    for k in ["meeting", "good", "morning", "good morning", "yes", "ok", "think", "decide", "i (not letter)",
              "1", "2", "don't know", "remote"]:
        ents.append(Entry(key=k, file=f"{k}.json", frames=90, active_first=30, active_last=59))
    for ch in "abcdefghijklmnopqrstuvwxyz":
        ents.append(Entry(key=ch, file=f"{ch.upper()}.json", frames=60, active_first=0, active_last=14))
    return Lexicon(ents, fps=30.0, mode="active")
