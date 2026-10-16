"""Silero VAD loading.

Whisper-Streaming and SimulStreaming load Silero VAD with
``torch.hub.load('snakers4/silero-vad', 'silero_vad')``.  The same model ships
in the ``silero-vad`` pip package, which avoids a network call at run time; we
use the package when available and fall back to torch.hub otherwise.  A fresh
model is loaded for every iterator because the model is stateful (RNN state).
"""
from __future__ import annotations


def load_vad_model():
    try:
        from silero_vad import load_silero_vad

        return load_silero_vad()  # TorchScript model with reset_states()
    except Exception:  # pragma: no cover - fallback path
        import torch

        model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
        return model


def fixed_vad_iterator(threshold: float = 0.5, min_silence_ms: int = 500, speech_pad_ms: int = 100):
    """A FixedVADIterator with Whisper-Streaming's defaults (500 ms silence, 100 ms padding)."""
    from .third_party.silero_vad_iterator import FixedVADIterator

    return FixedVADIterator(load_vad_model(), threshold=threshold,
                            min_silence_duration_ms=min_silence_ms, speech_pad_ms=speech_pad_ms)
