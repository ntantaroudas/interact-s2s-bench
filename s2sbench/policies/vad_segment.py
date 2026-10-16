"""P2 -- pause-based (VAD-endpointed) utterance transcription.

An approximation of the CORTEX2 Mediation Gateway behaviour reported in D4
("chunked audio based on pauses"): a Silero VAD (Whisper-Streaming defaults:
threshold 0.5, 500 ms minimum silence, 100 ms padding) detects speech segments
on the incoming 40 ms packets; a segment is transcribed once, in one call,
when its end is detected.  Segments longer than ``max_utt`` seconds are cut
(no overlap) so that a single talker cannot block output indefinitely.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from ..types import Word
from ..vad import fixed_vad_iterator
from .base import SR, GrowBuffer, OnlineProcessor, clean_words


class VADSegmentProcessor(OnlineProcessor):
    def __init__(self, asr, threshold: float = 0.5, min_silence_ms: int = 500,
                 speech_pad_ms: int = 100, max_utt: float = 15.0, vad_factory=None):
        super().__init__()
        self._vad_factory = vad_factory or (lambda: fixed_vad_iterator(threshold, min_silence_ms, speech_pad_ms))
        self.asr = asr
        self.threshold = threshold
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms
        self.max_utt = float(max_utt)
        self.name = f"P2-vad(sil={min_silence_ms}ms,max={max_utt:g}s)"
        self.reset()

    def reset(self) -> None:
        self.vad = self._vad_factory()
        self.buf = GrowBuffer()
        self.cur_start: Optional[float] = None
        self.queue: List[Tuple[float, float]] = []
        self.boundaries = []
        self.asr_calls = 0
        self.asr_audio_sec = 0.0

    def insert_audio_chunk(self, audio: np.ndarray) -> None:
        self.buf.append(audio)
        r = self.vad(audio)  # sample indices relative to the start of the stream
        # FixedVADIterator returns {'start'}, {'end'} or {'start','end'} (a whole
        # segment that began and ended inside this packet); see its docstring.
        if r:
            if "start" in r and "end" in r:
                s, e = r["start"] / SR, r["end"] / SR
                if self.cur_start is not None:
                    s = min(s, self.cur_start)
                if e > s:
                    self.queue.append((s, e))
                self.cur_start = None
            elif "start" in r:
                if self.cur_start is None:
                    self.cur_start = r["start"] / SR
            elif "end" in r and self.cur_start is not None:
                e = r["end"] / SR
                if e > self.cur_start:
                    self.queue.append((self.cur_start, e))
                self.cur_start = None
        # forced cut of over-long utterances
        while self.cur_start is not None and self.buf.end_time - self.cur_start >= self.max_utt:
            self.queue.append((self.cur_start, self.cur_start + self.max_utt))
            self.cur_start += self.max_utt

    def has_pending(self) -> bool:
        return bool(self.queue)

    def _run(self, s: float, e: float) -> List[Word]:
        audio = self.buf.get(s, e)
        words = self.asr.transcribe(audio, offset=s) if len(audio) else []
        self.asr_calls += 1
        self.asr_audio_sec += len(audio) / SR
        self.boundaries.append(e)
        keep_from = self.cur_start if self.cur_start is not None else e
        if self.queue:
            keep_from = min(keep_from, self.queue[0][0])
        self.buf.drop_before(max(0.0, keep_from - 1.0))
        return clean_words(words)

    def process_iter(self) -> List[Word]:
        if not self.queue:
            return []
        s, e = self.queue.pop(0)
        return self._run(s, e)

    def finish(self) -> List[Word]:
        out: List[Word] = []
        if self.cur_start is not None and self.buf.end_time > self.cur_start:
            self.queue.append((self.cur_start, self.buf.end_time))
            self.cur_start = None
        while self.queue:
            s, e = self.queue.pop(0)
            out.extend(self._run(s, e))
        return out

    def warmup(self, audio: np.ndarray) -> None:
        self.asr.warmup(audio)
