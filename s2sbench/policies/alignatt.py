"""P4 -- AlignAtt attention-guided simultaneous policy (Simul-Whisper /
SimulStreaming; Wang et al., 2024; Machacek & Polak, 2025).

Adapter around the reference implementation ufal/SimulStreaming (pinned
commit), imported unchanged.  Settings follow its command-line defaults
(greedy decoding, ``frame_threshold=25``, 30 s audio buffer) except
``min_chunk`` which we set to 1.0 s for all policies (SimulStreaming's own
default is 1.2 s).  The end-of-word (CIF) checkpoint from Simul-Whisper is
used when one exists for the model; otherwise the last word of each update is
always withheld, as documented by SimulStreaming.  The Silero model is injected
into its Voice Activity Controller instead of loading it via torch.hub.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import List, Optional

import numpy as np

from ..types import Word
from ..vad import load_vad_model
from .base import clean_words
from .local_agreement import third_party_dir


def _import_simulstreaming():
    path = os.path.join(third_party_dir(), "SimulStreaming")
    if not os.path.isfile(os.path.join(path, "simulstreaming_whisper.py")):
        raise ImportError(f"SimulStreaming not found in {path}; run scripts/fetch_third_party.sh")
    if path not in sys.path:
        sys.path.insert(0, path)
    import simulstreaming_whisper as ssw  # type: ignore
    from simulstreaming.whisper.whisper_streaming.vac_online_processor import VACOnlineASRProcessor  # type: ignore

    for name in ("simulstreaming_whisper", "simul_whisper", "whisper_streaming",
                 "simulstreaming.whisper.simul_whisper.simul_whisper",
                 "simulstreaming.whisper.whisper_streaming.vac_online_processor"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return ssw, VACOnlineASRProcessor


def cif_checkpoint(model: str) -> Optional[str]:
    p = os.path.join(third_party_dir(), "simul_whisper", "cif_models", f"{model}.pt")
    return p if os.path.isfile(p) else None


class AlignAttProcessor:
    def __init__(self, model: str = "large-v2", model_dir: str = "models/openai", min_chunk: float = 1.0,
                 vac: bool = True, frame_threshold: int = 25, beams: int = 1, audio_max_len: float = 30.0):
        ssw, VAC = _import_simulstreaming()
        os.makedirs(model_dir, exist_ok=True)
        cif = cif_checkpoint(model)
        self.asr = ssw.SimulWhisperASR(
            language="en", model_path=os.path.join(model_dir, f"{model}.pt"), cif_ckpt_path=cif,
            frame_threshold=frame_threshold, audio_max_len=audio_max_len, audio_min_len=0.0,
            segment_length=min_chunk, beams=beams, task="transcribe",
            decoder_type="greedy" if beams == 1 else "beam", never_fire=False,
            init_prompt=None, static_init_prompt=None, max_context_tokens=None, logdir=None,
        )
        self.online = ssw.SimulWhisperOnline(self.asr)

        class _VAC(VAC):
            def __init__(self_inner, online_chunk_size, online, min_buffered_length=1):
                from simulstreaming.whisper.whisper_streaming.silero_vad_iterator import FixedVADIterator  # type: ignore

                self_inner.online_chunk_size = online_chunk_size
                self_inner.online = online
                self_inner.min_buffered_frames = int(min_buffered_length * self_inner.SAMPLING_RATE)
                self_inner.vac = FixedVADIterator(load_vad_model())
                self_inner.init()

        self.proc = _VAC(min_chunk, self.online) if vac else self.online
        self.vac = vac
        self.min_chunk = float(min_chunk)
        self.tick = 0.04 if vac else self.min_chunk
        self.cif = cif
        self.name = f"P4-alignatt(min={min_chunk:g}s,ft={frame_threshold}{',vac' if vac else ''})"
        self.boundaries: List[float] = []

    @staticmethod
    def _words(o) -> List[Word]:
        """Convert SimulStreaming's output.  Its 'words' are Whisper word pieces;
        a piece without a leading space continues the previous word (possibly
        emitted in an earlier update).  We keep pieces as emitted -- that is what
        a downstream text-to-sign service receives -- and flag continuations so
        transcript-level metrics can re-join them (see metrics.join_continuations)."""
        if not o or "words" not in o:
            return []
        out = []
        for w in o["words"]:
            txt = w["text"]
            out.append(Word(txt, float(w["start"]), float(w["end"]),
                            meta={"cont": True} if (txt and not txt[0].isspace()) else {}))
        return clean_words(out)

    def insert_audio_chunk(self, audio: np.ndarray) -> None:
        self.proc.insert_audio_chunk(audio)

    def process_iter(self) -> List[Word]:
        return self._words(self.proc.process_iter())

    def has_pending(self) -> bool:
        return False

    def finish(self) -> List[Word]:
        return self._words(self.proc.finish())

    def warmup(self, audio: np.ndarray) -> None:
        self.asr.warmup(audio[:16000])
        self.online.init()

    def reset(self) -> None:
        self.proc.init()

    def stats(self) -> dict:
        return {"cif": bool(self.cif)}
