"""Text-to-sign mapping variants (what the avatar is asked to play).

M0  "deployed"  -- the behaviour described for INTERACT's text-to-sign endpoint:
     each whitespace-separated word is lower-cased and stripped of surrounding
     punctuation; if the result is a lexicon key it is signed, otherwise it is
     fingerspelled letter by letter (digits 1-9 use number signs, 0 uses O).
     Multi-word lexicon entries can never be reached this way.

M1  "reduced"   -- a load-reduction heuristic, NOT a linguistic translation:
     greedy longest-match of multi-word entries within a commit batch, a small
     transparent alias table, lemma fallback (simplemma), omission of
     function words that have no lexicon entry, clitic splitting; remaining
     out-of-vocabulary (OOV) words are fingerspelled.

M2  "no-fs"     -- as M1, but OOV words are not fingerspelled; they are left to
     the caption channel (counted as ``caption_only``).

All variants map a *batch* of words that a streaming policy committed together;
multi-word matching never waits for future words, so mapping adds no latency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import groupby
from typing import Dict, List, Optional, Sequence

from .lexicon import Lexicon
from .normalize import FILLERS, strip_for_lookup, norm_token
from .types import Word

# --- M1 resources (kept deliberately small and reported in the paper) --------
ALIASES: Dict[str, str] = {
    "okay": "ok", "o.k.": "ok", "k": "ok",
    "yeah": "yes", "yep": "yes", "yup": "yes", "nope": "no",
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "i": "i (not letter)", "e-mail": "email",
}
FUNCTION_WORDS = frozenset(
    """a an the
    am is are was were be been being 's 're 'm 've 'll 'd
    do does did doing has had having will would shall should could might
    of to in on at for with by from about as into onto than via per
    and or so if because although though whether nor
    it its it's we us our ours them their theirs your yours his her hers him
    that this's these those there here
    just well really very quite actually basically
    oh ah eh huh hey""".split()
)
CLITICS = ("n't", "'s", "'re", "'m", "'ve", "'ll", "'d")
IRREGULAR_NEG = {"can't": ("can", "not"), "won't": ("will", "not"), "shan't": ("shall", "not"),
                 "ain't": ("is", "not"), "cannot": ("can", "not")}


@dataclass
class SignItem:
    kind: str                 # "LEX" | "FS" | "CAP" (caption only, not played)
    label: str                # lexicon key, or the string being spelled/captioned
    clips: List[str]          # clip keys played in order (empty for CAP)
    duration: float           # seconds at playback rate 1.0
    src: List[int]            # indices into the input word list
    emit: Optional[float]     # when the item is available to the avatar
    ts_end: float             # ASR/reference end time of the last source word
    ts_start: float = 0.0

    @property
    def token(self) -> str:
        """Single whitespace-free token used for sign-sequence alignment."""
        return {"LEX": "L:", "FS": "F:", "CAP": "C:"}[self.kind] + "_".join(self.label.split())

    @property
    def played(self) -> bool:
        return self.kind in ("LEX", "FS")


class _Base:
    name = "base"

    def __init__(self, lexicon: Lexicon):
        self.lx = lexicon

    # helpers --------------------------------------------------------------
    def _lex(self, key: str, src, words) -> SignItem:
        return SignItem("LEX", key, [key], self.lx.duration(key), list(src),
                        _emit(words, src), words[src[-1]].end, words[src[0]].start)

    def _fs(self, s: str, src, words) -> Optional[SignItem]:
        clips = [k for k in (self.lx.letter_key(c) for c in s) if k is not None]
        if not clips:
            return None
        return SignItem("FS", s, clips, sum(self.lx.duration(k) for k in clips), list(src),
                        _emit(words, src), words[src[-1]].end, words[src[0]].start)

    def _cap(self, s: str, src, words) -> SignItem:
        return SignItem("CAP", s, [], 0.0, list(src), _emit(words, src),
                        words[src[-1]].end, words[src[0]].start)

    def map(self, words: Sequence[Word], batch_by_emit: bool = True) -> List[SignItem]:
        """Map words to sign items.  Batches are runs of words with the same
        ``emit`` time (one commit); reference words (emit=None) form one batch."""
        items: List[SignItem] = []
        idx = list(range(len(words)))
        if batch_by_emit:
            groups = [list(g) for _, g in groupby(idx, key=lambda i: words[i].emit)]
        else:
            groups = [idx]
        for g in groups:
            items.extend(self._map_batch(words, g))
        return items

    def _map_batch(self, words, idx) -> List[SignItem]:  # pragma: no cover
        raise NotImplementedError


def _emit(words, src) -> Optional[float]:
    e = [words[i].emit for i in src if words[i].emit is not None]
    return max(e) if e else None


class DeployedMapper(_Base):
    """M0."""

    name = "M0"

    def _map_batch(self, words, idx):
        out = []
        for i in idx:
            t = strip_for_lookup(words[i].text)
            if not t:
                continue
            if t in self.lx:
                out.append(self._lex(t, [i], words))
            else:
                it = self._fs(t, [i], words)
                if it is not None:
                    out.append(it)
        return out


class ReducedMapper(_Base):
    """M1 (fingerspell=True) and M2 (fingerspell=False)."""

    def __init__(self, lexicon: Lexicon, fingerspell: bool = True):
        super().__init__(lexicon)
        self.fingerspell = fingerspell
        self.name = "M1" if fingerspell else "M2"
        # phrase index: normalised multi-word key -> lexicon key
        self.phrases: Dict[tuple, str] = {}
        for k in lexicon.by_key:
            toks = tuple(k.strip().split())
            if len(toks) > 1 and not any(c in k for c in "()"):
                self.phrases[toks] = k
        self.max_n = max((len(p) for p in self.phrases), default=1)
        try:
            import simplemma

            self._lemma = lambda w: simplemma.lemmatize(w, lang="en")
        except Exception:  # pragma: no cover - optional dependency
            self._lemma = lambda w: w

    def _lookup(self, tok: str) -> Optional[str]:
        # aliases first: e.g. the pronoun "i" must not be signed as the letter I
        for cand in (ALIASES.get(tok), tok, norm_token(tok), ALIASES.get(norm_token(tok))):
            if cand and cand in self.lx:
                return cand
        lem = self._lemma(tok)
        if lem in self.lx:
            return lem
        if ALIASES.get(lem, "") in self.lx:
            return ALIASES[lem]
        return None

    def _map_batch(self, words, idx):
        # word-level tokens (lower-cased, punctuation stripped; apostrophes kept)
        toks = []  # (token, word_index)
        for i in idx:
            t = strip_for_lookup(words[i].text)
            if not t or t in FILLERS:
                continue
            for p in t.replace("-", " ").split():
                if p and p not in FILLERS:
                    toks.append((p, i))
        out: List[SignItem] = []
        j = 0
        while j < len(toks):
            # 1) greedy multi-word match within the batch
            matched = False
            for n in range(min(self.max_n, len(toks) - j), 1, -1):
                key = self.phrases.get(tuple(t for t, _ in toks[j:j + n]))
                if key is not None:
                    src = sorted({wi for _, wi in toks[j:j + n]})
                    out.append(self._lex(key, src, words))
                    j += n
                    matched = True
                    break
            if matched:
                continue
            tok, wi = toks[j]
            j += 1
            out.extend(self._map_token(tok, wi, words))
        return out

    def _map_token(self, tok, wi, words) -> List[SignItem]:
        if tok.startswith("'"):
            # clitic parts ('s, 're, 'm, 've, 'll, 'd) are function words; they must not reach the lookup,
            # which strips the apostrophe and would sign the letters S, M or D
            return []
        key = self._lookup(tok)
        if key is not None:
            return [self._lex(key, [wi], words)]
        if tok in IRREGULAR_NEG:
            res = []
            for part in IRREGULAR_NEG[tok]:
                res.extend(self._map_token(part, wi, words))
            return res
        # clitics: "don't" -> do + not ; "it's" -> it + 's
        for c in CLITICS:
            if tok.endswith(c) and len(tok) > len(c):
                host, cl = tok[: -len(c)], ("not" if c == "n't" else c)
                res = []
                for part in (host, cl):
                    res.extend(self._map_token(part, wi, words))
                return res
        if tok in FUNCTION_WORDS:
            return []
        if self.fingerspell:
            it = self._fs(tok, [wi], words)
            return [it] if it is not None else []
        return [self._cap(tok, [wi], words)]


def make_mapper(name: str, lexicon: Lexicon) -> _Base:
    name = name.upper()
    if name == "M0":
        return DeployedMapper(lexicon)
    if name == "M1":
        return ReducedMapper(lexicon, fingerspell=True)
    if name == "M2":
        return ReducedMapper(lexicon, fingerspell=False)
    raise ValueError(f"unknown mapper {name}")
