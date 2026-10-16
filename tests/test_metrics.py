import pytest

from s2sbench.mapper import make_mapper
from s2sbench.metrics import join_continuations, oracle_metrics, sign_metrics, transcript_metrics
from s2sbench.stats import holm
from s2sbench.types import Word


def mk(seq, emit_delay=None):
    out, t = [], 0.0
    for tok in seq:
        w = Word(tok, t, t + 0.4)
        if emit_delay is not None:
            w.emit = w.end + emit_delay
        out.append(w)
        t += 0.5
    return out


def test_transcript_metrics_basic():
    ref = mk(["good", "morning", "the", "meeting", "starts", "now"])
    hyp = mk(["good", "morning", "the", "meet", "starts", "now", "now"], emit_delay=1.0)
    m = transcript_metrics(ref, hyp, duration=10.0)
    assert m["wer_sub"] == 1 and m["wer_ins"] == 1
    assert m["frag_per_1k"] == pytest.approx(1000 / 6)
    assert m["dup_per_1k"] == pytest.approx(1000 / 6)
    assert m["emit_lag_median"] == pytest.approx(1.0, abs=0.11)


def test_join_continuations():
    ws = [Word("meet", 0, .3, emit=1), Word("ing", .3, .5, emit=2, meta={"cont": True}), Word("now", .6, .9, emit=2)]
    j = join_continuations(ws)
    assert [w.text for w in j] == ["meeting", "now"] and j[0].emit == 2


def test_sign_metrics_and_oracle(lexicon):
    ref = mk(["good", "morning", "the", "meeting"])
    hyp = mk(["good", "morning", "the", "meet"], emit_delay=0.5)
    mp = make_mapper("M0", lexicon)
    s = sign_metrics(mp, ref, hyp, duration=10.0)
    assert s["ser"] == pytest.approx(1 / 4)              # L:meeting -> F:meet
    assert s["spurious_fs_per_min"] == pytest.approx(1 / (10 / 60))
    o, played = oracle_metrics(mp, ref, duration=10.0)
    assert o["fs_items"] == 1 and o["lex_items"] == 3


def test_holm():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_multiword_sign_tokens_align(lexicon):
    ref = mk(["good", "morning", "meeting"])
    hyp = mk(["good", "morning", "meeting"])
    for w in hyp:
        w.emit = 2.0                       # one commit batch -> phrase can be matched
    mp = make_mapper("M1", lexicon)
    s = sign_metrics(mp, ref, hyp, duration=10.0)
    assert s["n_ref_items"] == 2 and s["ser"] == 0.0 and s["delivered"] == 1.0
    # committed in separate batches, the phrase is signed as two single signs
    s2 = sign_metrics(mp, ref, mk(["good", "morning", "meeting"], emit_delay=0.3), duration=10.0)
    assert s2["ser"] == 1.0


def test_load_reference_excerpt(tmp_path):
    """Excerpt selection keeps words by midpoint, shifts times and rebuilds speaker segments."""
    import json

    from s2sbench.io import load_reference

    words = [
        {"text": "hello", "start": 9.0, "end": 9.8, "speaker": "A", "meta": {"seg": "A:0"}},      # before
        {"text": "there", "start": 9.9, "end": 10.3, "speaker": "A", "meta": {"seg": "A:0"}},     # mid 10.1: in
        {"text": "um", "start": 11.0, "end": 11.2, "speaker": "B", "meta": {"seg": "B:0"}},       # filler
        {"text": "world", "start": 12.0, "end": 12.5, "speaker": "B", "meta": {"seg": "B:0"}},    # in
        {"text": "bye", "start": 14.8, "end": 15.6, "speaker": "A", "meta": {"seg": "A:1"}},      # mid 15.2: out
    ]
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"words": words, "segments": [
        {"speaker": "A", "start": 9.0, "end": 10.3, "text": "hello there"},
        {"speaker": "B", "start": 11.0, "end": 12.5, "text": "um world"},
        {"speaker": "A", "start": 14.8, "end": 15.6, "text": "bye"}]}))
    ref, segs = load_reference(str(p), limit=5.0, start=10.0)
    assert [w.text for w in ref] == ["there", "world"]
    assert abs(ref[0].start - (-0.1)) < 1e-9 and abs(ref[1].end - 2.5) < 1e-9
    assert [s["text"] for s in segs] == ["there", "um world"]       # rebuilt from the kept words
    assert abs(segs[1]["start"] - 1.0) < 1e-9
    # without start, the old behaviour (first `limit` seconds) is unchanged
    ref0, segs0 = load_reference(str(p), limit=10.0)
    assert [w.text for w in ref0] == ["hello"] and len(segs0) == 1


def test_prepare_for_mapping_joins_pieces_and_drops_annotations():
    from s2sbench.metrics import prepare_for_mapping
    from s2sbench.types import Word

    ws = [Word("We", 0.0, 0.2, emit=1.0), Word("meet", 0.2, 0.4, emit=1.0), Word("ing", 0.4, 0.6, emit=2.0, meta={"cont": True}),
          Word("[BLANK", 0.7, 0.8, emit=2.0), Word("_AUDIO]", 0.8, 0.9, emit=2.0, meta={"cont": True}),
          Word("(keyboard", 1.0, 1.1, emit=3.0), Word("clicking)", 1.1, 1.2, emit=3.0), Word("now", 1.3, 1.5, emit=3.0)]
    out = prepare_for_mapping(ws)
    assert [w.text for w in out] == ["We", "meeting", "now"]
    assert out[1].emit == 2.0          # a continuation piece delays its word


def test_drop_annotations_without_closing_brackets():
    """Word-piece output can lose the closing bracket; the span then ends at the commit or case change."""
    from s2sbench.metrics import drop_annotations
    from s2sbench.types import Word

    def W(t, e):
        return Word(t, 0.0, 0.1, emit=e)

    ws = [W("(laughing", 1.0), W("For", 1.0), W("the", 1.0), W("[INTERPOSING", 2.0), W("VOICES", 2.0),
          W("Found", 3.0), W("(keyboard", 4.0), W("clacking", 4.0), W("Now", 5.0), W("(laughing", 6.0), W("is", 7.0)]
    assert [w.text for w in drop_annotations(ws)] == ["For", "the", "Found", "Now", "is"]
