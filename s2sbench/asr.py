"""ASR back-ends used by the policies we implement ourselves (P1, P2).

``FasterWhisperBackend`` wraps faster-whisper (CTranslate2).  Decoding
settings follow the reference Whisper-Streaming faster-whisper back-end
(beam_size=5, word timestamps) so that P1/P2/P3 differ only in their
segmentation/commit policy, not in decoding.

``OracleASR`` returns reference words that fall inside the audio window and is
used only by unit tests (it lets us test policies without a model).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from .types import Word

SR = 16000


class FasterWhisperBackend:
    def __init__(
        self,
        model: str = "large-v2",
        device: str = "cuda",
        compute_type: str = "float16",
        language: str = "en",
        beam_size: int = 5,
        download_root: Optional[str] = None,
        condition_on_previous_text: bool = False,
    ):
        from faster_whisper import WhisperModel

        self.model_name = model
        self.language = language
        self.beam_size = beam_size
        self.condition_on_previous_text = condition_on_previous_text
        self.model = WhisperModel(model, device=device, compute_type=compute_type, download_root=download_root)

    def transcribe(self, audio: np.ndarray, offset: float = 0.0, prompt: Optional[str] = None) -> List[Word]:
        """Transcribe one window; returns words with absolute times (offset added)."""
        if audio.size == 0:
            return []
        segments, _ = self.model.transcribe(
            audio.astype(np.float32),
            language=self.language,
            beam_size=self.beam_size,
            word_timestamps=True,
            condition_on_previous_text=self.condition_on_previous_text,
            initial_prompt=prompt,
            vad_filter=False,
        )
        out: List[Word] = []
        for seg in segments:  # the generator runs the decoder lazily
            for w in seg.words or []:
                t = w.word.strip()
                if t:
                    out.append(Word(t, offset + float(w.start), offset + float(w.end),
                                    meta={"p": round(float(w.probability), 4)}))
        return out

    def warmup(self, audio: np.ndarray) -> None:
        self.transcribe(audio[: SR])


class OracleASR:
    """Test double: 'recognises' reference words inside a window.

    A word cut by a window edge is returned as a fragment whose length is
    proportional to the part of the word inside the window -- a simple model of
    the word-fragmentation failure mode of fixed-length chunking.
    """

    def __init__(self, reference: Sequence[Word]):
        self.ref = list(reference)
        self.calls = 0

    def transcribe(self, audio: np.ndarray, offset: float = 0.0, prompt: Optional[str] = None) -> List[Word]:
        self.calls += 1
        a, b = offset, offset + len(audio) / SR
        out: List[Word] = []
        for w in self.ref:
            if w.end <= a or w.start >= b:
                continue
            s, e = max(w.start, a), min(w.end, b)
            frac = (e - s) / max(w.end - w.start, 1e-6)
            if frac >= 0.999:
                out.append(Word(w.text, w.start, w.end))
            elif frac >= 0.25:
                n = max(1, int(round(len(w.text) * frac)))
                frag = w.text[:n] if s == w.start else w.text[-n:]
                out.append(Word(frag, s, e, meta={"fragment": True}))
        return out

    def warmup(self, audio: np.ndarray) -> None:
        return None
