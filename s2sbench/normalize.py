"""Text normalisation.

Two normalisers are used, for two different purposes:

* ``wer_normalize`` -- OpenAI Whisper's ``EnglishTextNormalizer`` (vendored,
  MIT licence) applied to whole transcripts.  This is the de-facto standard for
  scoring Whisper output and is used only for the headline WER.
* ``norm_token`` / ``tokens_of`` -- a light, *per-word* normaliser that keeps a
  one-to-many mapping between a timestamped word and its tokens.  It is used
  for every time-aware analysis (latency, fragmentation, hallucination) and for
  aligning sign sequences, where we must not merge or reorder words.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from functools import lru_cache
from typing import Iterable, List, Tuple

from .types import Word

_HERE = os.path.dirname(__file__)
with open(os.path.join(_HERE, "third_party", "whisper_normalizers", "english.json"), encoding="utf-8") as _f:
    _SPELLING = json.load(_f)  # British -> American spelling map shipped with Whisper

# Hesitations/back-channels that are not lexical content.  The first six are the
# ones Whisper's EnglishTextNormalizer removes; the rest are AMI spellings of
# the same phenomena.  They are removed from both reference and hypothesis
# before any per-token analysis.
FILLERS = frozenset(
    {"hmm", "mm", "mhm", "mmm", "uh", "um", "uhm", "hm", "er", "erm", "mm-hmm", "mm-mm", "uh-huh", "uh-uh"}
)

_STRIP = "\"'`’‘“”.,;:!?()[]{}<>«»…–—-_/\\*~^|"
_APOS = str.maketrans({"’": "'", "‘": "'", "`": "'"})


@lru_cache(maxsize=1)
def _english_normalizer():
    from .third_party.whisper_normalizers import EnglishTextNormalizer

    return EnglishTextNormalizer()


def wer_normalize(text: str) -> str:
    """Whisper EnglishTextNormalizer on a whole transcript."""
    return _english_normalizer()(text)


def norm_token(tok: str) -> str:
    """Lower-case, NFKC, unify apostrophes, strip edge punctuation, US spelling."""
    s = unicodedata.normalize("NFKC", tok).translate(_APOS).lower().strip()
    s = s.strip(_STRIP)
    return _SPELLING.get(s, s)


def tokens_of(text: str) -> List[str]:
    """Split one (possibly hyphenated) word into normalised, non-filler tokens."""
    raw = norm_token(text)
    if not raw or raw in FILLERS:
        return []
    parts = [p for p in re.split(r"[\s\-]+", raw) if p]
    out = []
    for p in parts:
        p = norm_token(p)
        if p and p not in FILLERS:
            out.append(p)
    return out


def expand_words(words: Iterable[Word]) -> Tuple[List[str], List[int]]:
    """Tokenise a word list; returns (tokens, index of source word per token)."""
    toks: List[str] = []
    src: List[int] = []
    for i, w in enumerate(words):
        for t in tokens_of(w.text):
            toks.append(t)
            src.append(i)
    return toks, src


def strip_for_lookup(tok: str) -> str:
    """The minimal clean-up assumed for the deployed text-to-sign endpoint:
    lower-case and strip surrounding punctuation (no spelling or lemma changes)."""
    s = unicodedata.normalize("NFKC", tok).translate(_APOS).lower().strip()
    return s.strip(_STRIP)
