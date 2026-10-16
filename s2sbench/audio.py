"""Audio loading (16 kHz mono float32)."""
from __future__ import annotations

import numpy as np

SR = 16000


def load_audio(path: str, start: float = 0.0, end: float | None = None) -> np.ndarray:
    import soundfile as sf

    info = sf.info(path)
    i0 = int(start * info.samplerate)
    i1 = None if end is None else int(end * info.samplerate)
    x, sr = sf.read(path, start=i0, stop=i1, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        import librosa

        x = librosa.resample(x, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(x, dtype=np.float32)
