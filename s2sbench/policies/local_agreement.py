"""P3 -- LocalAgreement-2 (Whisper-Streaming; Machacek, Dabre & Bojar, 2023).

Thin adapter around the reference implementation (ufal/whisper_streaming,
``whisper_online.py`` at the pinned commit), imported unchanged.  We only:

* subclass its faster-whisper back-end to choose device/compute type (the
  original hard-codes CUDA float16, which is also what we use on the T4);
* record the newly committed words (with their timestamps) after each
  ``process_iter``/``finish`` call -- the original returns them concatenated;
* inject the Silero model into its Voice Activity Controller instead of
  calling ``torch.hub`` at run time (same model);
* clear the already-flushed hypothesis after ``finish()`` so that the final
  end-of-stream flush does not repeat words already flushed at the end of the
  last voice segment (a duplication present in the reference simulation).

Note: at the end of a voice segment (VAC) and at the end of the stream the
reference flushes its last unconfirmed hypothesis without decoding the audio
received since the previous update; ``final_decode=True`` (ablation) runs one
more update first.

Decoding settings are the original's: beam 5, word timestamps,
``condition_on_previous_text=True`` with a 200-character prompt of committed
text, buffer trimming at completed segments when the buffer exceeds 15 s.
"""
from __future__ import annotations

import io
import os
import sys
from typing import List

import numpy as np

from ..types import Word
from ..vad import load_vad_model
from .base import clean_words


def third_party_dir() -> str:
    return os.environ.get(
        "S2S_THIRD_PARTY",
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "third_party"),
    )


def _import_whisper_online():
    path = os.path.join(third_party_dir(), "whisper_streaming")
    if not os.path.isfile(os.path.join(path, "whisper_online.py")):
        raise ImportError(f"whisper_streaming not found in {path}; run scripts/fetch_third_party.sh")
    if path not in sys.path:
        sys.path.insert(0, path)
    import whisper_online  # type: ignore

    return whisper_online


class LocalAgreementProcessor:
    """Online processor exposing insert_audio_chunk/process_iter/finish."""

    def __init__(self, model: str = "large-v2", device: str = "cuda", compute_type: str = "float16",
                 min_chunk: float = 1.0, vac: bool = True, buffer_trimming_sec: float = 15.0,
                 download_root: str | None = None, final_decode: bool = False):
        wo = _import_whisper_online()
        self._devnull = io.StringIO()

        class _ASR(wo.FasterWhisperASR):
            def load_model(self_inner, modelsize=None, cache_dir=None, model_dir=None):
                from faster_whisper import WhisperModel

                return WhisperModel(model_dir or modelsize, device=device, compute_type=compute_type,
                                    download_root=cache_dir)

        class _Tracking(wo.OnlineASRProcessor):
            def __init__(self_inner, *a, **kw):
                self_inner.emitted = []
                super().__init__(*a, **kw)

            def process_iter(self_inner):
                n0 = len(self_inner.commited)
                ret = super().process_iter()
                self_inner.emitted.extend(self_inner.commited[n0:])
                return ret

            def finish(self_inner):
                if final_decode and len(self_inner.audio_buffer):
                    # optional variant: decode the audio received since the last update
                    # before flushing (the reference flushes the last hypothesis only)
                    self_inner.process_iter()
                rest = list(self_inner.transcript_buffer.complete())
                ret = super().finish()
                self_inner.emitted.extend(rest)
                # The reference finish() leaves the flushed hypothesis in the buffer, so a
                # second finish() (VAC end-of-voice followed by end-of-stream) would emit it
                # again.  Clearing it is behaviour-neutral otherwise: the next voice segment
                # re-initialises the processor.
                self_inner.transcript_buffer.buffer = []
                return ret

        class _VAC(wo.VACOnlineASRProcessor):
            def __init__(self_inner, online_chunk_size, online):
                from s2sbench.third_party.silero_vad_iterator import FixedVADIterator

                self_inner.online_chunk_size = online_chunk_size
                self_inner.online = online
                self_inner.vac = FixedVADIterator(load_vad_model())  # 500 ms silence, 100 ms pad
                self_inner.logfile = online.logfile
                self_inner.init()

        self.asr = _ASR(lan="en", modelsize=model, cache_dir=download_root, logfile=self._devnull)
        self.inner = _Tracking(self.asr, tokenizer=None, buffer_trimming=("segment", buffer_trimming_sec),
                               logfile=self._devnull)
        self.proc = _VAC(min_chunk, self.inner) if vac else self.inner
        self.vac = vac
        self.min_chunk = float(min_chunk)
        self.tick = 0.04 if vac else self.min_chunk
        self.name = f"P3-localagreement(min={min_chunk:g}s{',vac' if vac else ''}{',final-decode' if final_decode else ''})"
        self.boundaries: List[float] = []

    # -- interface ------------------------------------------------------------
    def _drain(self) -> List[Word]:
        out = [Word(t, float(b), float(e)) for (b, e, t) in self.inner.emitted]
        self.inner.emitted = []
        self._devnull.seek(0)
        self._devnull.truncate(0)
        return clean_words(out)

    def insert_audio_chunk(self, audio: np.ndarray) -> None:
        self.proc.insert_audio_chunk(audio)

    def process_iter(self) -> List[Word]:
        self.proc.process_iter()
        return self._drain()

    def has_pending(self) -> bool:
        return False

    def finish(self) -> List[Word]:
        self.proc.finish()
        return self._drain()

    def warmup(self, audio: np.ndarray) -> None:
        self.asr.transcribe(audio[:16000])

    def reset(self) -> None:
        self.proc.init()
        self.inner.emitted = []

    def stats(self) -> dict:
        return {}
