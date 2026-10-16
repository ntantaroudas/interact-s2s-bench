import numpy as np

from s2sbench.asr import OracleASR
from s2sbench.policies.fixed import FixedChunkProcessor
from s2sbench.policies.vad_segment import VADSegmentProcessor
from s2sbench.simulate import simulate
from s2sbench.types import Word

SR = 16000


def ref_words():
    # 10 words of 0.6 s separated by 0.1 s, starting at 0.25 s
    out, t = [], 0.25
    for i in range(10):
        out.append(Word(f"word{i}", t, t + 0.6))
        t += 0.7
    return out


def test_fixed_with_overlap_keeps_each_word_once():
    ref = ref_words()
    audio = np.zeros(int(8.0 * SR), dtype=np.float32)
    proc = FixedChunkProcessor(OracleASR(ref), hop=1.0, context=0.5)
    res = simulate(proc, audio, comp_aware=False)
    texts = [w.text for w in res.words]
    # every word whose midpoint is inside a core window is kept exactly once
    assert sorted(set(texts)) == sorted(texts)
    assert len(texts) == 10
    # emission time of window k is (k+1)*hop + context (computation-unaware)
    first = res.words[0]
    assert first.text == "word0" and abs(first.emit - 1.5) < 0.05


def test_fixed_without_overlap_fragments_words():
    ref = ref_words()
    audio = np.zeros(int(8.0 * SR), dtype=np.float32)
    res = simulate(FixedChunkProcessor(OracleASR(ref), hop=1.0, context=0.0), audio, comp_aware=False)
    frags = [w for w in res.words if not w.text.startswith("word") or len(w.text) < 5]
    assert len(frags) > 0          # words cut by 1 s boundaries come out as fragments


class FakeVAD:
    """Speech from 0.25 s to 3.0 s and from 5.0 s to 6.5 s, reported with a delay."""

    def __init__(self):
        self.t = 0.0
        self.events = [(0.30, {"start": int(0.25 * SR)}), (3.6, {"end": int(3.0 * SR)}),
                       (5.05, {"start": int(5.0 * SR)}), (7.1, {"end": int(6.5 * SR)})]

    def __call__(self, x):
        self.t += len(x) / SR
        out = {}
        while self.events and self.events[0][0] <= self.t:
            out.update(self.events.pop(0)[1])
        return out or None


def test_vad_segments_emit_after_endpoint():
    ref = ref_words()
    audio = np.zeros(int(8.0 * SR), dtype=np.float32)
    proc = VADSegmentProcessor(OracleASR(ref), vad_factory=FakeVAD)
    res = simulate(proc, audio, comp_aware=False)
    assert proc.boundaries[:2] == [3.0, 6.5]
    # words of the first segment are emitted once the end is detected (~3.6 s)
    w0 = [w for w in res.words if w.text == "word0"][0]
    assert 3.55 <= w0.emit <= 3.7


def test_comp_aware_clock_advances_with_compute():
    class SlowASR(OracleASR):
        def transcribe(self, audio, offset=0.0, prompt=None):
            import time
            time.sleep(0.02)
            return super().transcribe(audio, offset, prompt)

    ref = ref_words()
    audio = np.zeros(int(4.0 * SR), dtype=np.float32)
    res = simulate(FixedChunkProcessor(SlowASR(ref), hop=1.0, context=0.0), audio, comp_aware=True)
    assert res.compute_total >= 0.02 * 4
    assert all(w.emit >= w.end for w in res.words)
