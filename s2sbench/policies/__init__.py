"""Policy factory.

Policy identifiers used in configs and in the paper:

    P1  fixed           fixed-length windows (INTERACT baseline)
    P2  vad             pause/VAD-endpointed utterances (Mediation-Gateway-like)
    P3  localagreement  LocalAgreement-2, Whisper-Streaming reference implementation
    P4  alignatt        AlignAtt, SimulStreaming reference implementation
"""
from __future__ import annotations

from typing import Any, Dict


def build_policy(cfg: Dict[str, Any], asr=None):
    """Create an online processor from a config dict.

    ``asr`` may be passed to share one loaded faster-whisper model between P1/P2
    runs (P3/P4 load their own model through the reference implementations).
    """
    kind = cfg["policy"].lower()
    model = cfg.get("model", "large-v2")
    device = cfg.get("device", "cuda")
    compute_type = cfg.get("compute_type", "float16" if device == "cuda" else "int8")
    if kind in ("fixed", "p1", "vad", "p2"):
        if asr is None:
            from ..asr import FasterWhisperBackend

            asr = FasterWhisperBackend(model=model, device=device, compute_type=compute_type,
                                       beam_size=cfg.get("beam_size", 5),
                                       download_root=cfg.get("download_root"))
        if kind in ("fixed", "p1"):
            from .fixed import FixedChunkProcessor

            return FixedChunkProcessor(asr, hop=cfg.get("hop", 1.0), context=cfg.get("context", 0.25))
        from .vad_segment import VADSegmentProcessor

        return VADSegmentProcessor(asr, threshold=cfg.get("vad_threshold", 0.5),
                                   min_silence_ms=cfg.get("min_silence_ms", 500),
                                   speech_pad_ms=cfg.get("speech_pad_ms", 100),
                                   max_utt=cfg.get("max_utt", 15.0))
    if kind in ("localagreement", "p3"):
        from .local_agreement import LocalAgreementProcessor

        return LocalAgreementProcessor(model=model, device=device, compute_type=compute_type,
                                       min_chunk=cfg.get("min_chunk", 1.0), vac=cfg.get("vac", True),
                                       buffer_trimming_sec=cfg.get("buffer_trimming_sec", 15.0),
                                       download_root=cfg.get("download_root"),
                                       final_decode=cfg.get("final_decode", False))
    if kind in ("alignatt", "p4"):
        from .alignatt import AlignAttProcessor

        return AlignAttProcessor(model=model, model_dir=cfg.get("openai_model_dir", "models/openai"),
                                 min_chunk=cfg.get("min_chunk", 1.0), vac=cfg.get("vac", True),
                                 frame_threshold=cfg.get("frame_threshold", 25), beams=cfg.get("beams", 1),
                                 audio_max_len=cfg.get("audio_max_len", 30.0))
    raise ValueError(f"unknown policy {kind}")
