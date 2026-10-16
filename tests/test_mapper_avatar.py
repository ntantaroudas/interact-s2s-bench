import pytest

from s2sbench.avatar import demand_ratio, schedule
from s2sbench.lexicon import Entry, Lexicon
from s2sbench.mapper import make_mapper
from s2sbench.types import Word


def W(text, s, e, emit=None):
    return Word(text, s, e, emit=emit)


def test_lexicon_durations_and_collisions():
    ents = [Entry("wait", "Wait.json", 216, 8, 49), Entry("wait", "wait.json", 65, 19, 51)]
    lx = Lexicon(ents, fps=30, mode="active")
    assert lx.collisions == {"wait": ["Wait.json", "wait.json"]}
    assert lx.by_key["wait"].file == "wait.json"          # last write wins (sorted order)
    assert lx.duration("wait") == pytest.approx(33 / 30)
    full = Lexicon(ents, fps=30, mode="full")
    assert full.duration("wait") == pytest.approx(65 / 30)


def test_m0_deployed(lexicon):
    words = [W("Meeting,", 0, 0.5, 1), W("the", 0.5, 0.6, 1), W("Good", 0.6, 0.8, 1), W("morning.", 0.8, 1.2, 1)]
    items = make_mapper("M0", lexicon).map(words)
    assert [i.token for i in items] == ["L:meeting", "F:the", "L:good", "L:morning"]
    assert items[1].clips == ["t", "h", "e"]
    assert items[1].duration == pytest.approx(3 * 0.5)
    # pronoun "I" is looked up as the letter key "i" in the deployed scheme
    it = make_mapper("M0", lexicon).map([W("I", 0, 0.2, 1)])
    assert it[0].token == "F:i" or it[0].token == "L:i"


def test_m1_phrases_aliases_lemmas_function_words(lexicon):
    words = [W("Good", 0, .2, 1), W("morning", .2, .5, 1), W("the", .5, .6, 1), W("meetings", .6, 1, 1),
             W("yeah", 1, 1.2, 1), W("I", 1.2, 1.3, 1), W("decided", 1.3, 1.6, 1), W("don't", 1.6, 1.8, 1),
             W("know", 1.8, 2.0, 1), W("Zork", 2.0, 2.3, 1)]
    toks = [i.token for i in make_mapper("M1", lexicon).map(words)]
    assert toks == ["L:good_morning", "L:meeting", "L:yes", "L:i_(not_letter)", "L:decide", "L:don't_know", "F:zork"]
    toks2 = [i.token for i in make_mapper("M2", lexicon).map(words)]
    assert toks2[-1] == "C:zork"


def test_m1_phrase_not_across_batches(lexicon):
    words = [W("good", 0, .2, emit=1.0), W("morning", .2, .5, emit=2.0)]
    toks = [i.token for i in make_mapper("M1", lexicon).map(words)]
    assert toks == ["L:good", "L:morning"]


def test_schedule_fifo_and_bounded(lexicon):
    words = [W("meeting", 0, .5, 0.5), W("yes", .5, .7, 0.7), W("ok", .7, .9, 0.9)]
    items = make_mapper("M0", lexicon).map(words)
    p = schedule(items, rate=1.0)
    assert [round(x.start, 3) for x in p] == [0.5, 1.5, 2.5]
    p2 = schedule(items, rate=2.0)
    assert [round(x.start, 3) for x in p2] == [0.5, 1.0, 1.5]
    pb = schedule(items, rate=1.0, policy="bounded", max_lag=1.0)
    assert [x.dropped for x in pb] == [False, False, True]   # 2.5 - 0.9 = 1.6 > 1.0
    assert demand_ratio(items, 3.0) == pytest.approx(1.0)


def test_m1_clitics_are_not_signed_as_letters(lexicon):
    """'it's' -> IT (function word, omitted) + 's (clitic, omitted); never the letter S."""
    from s2sbench.mapper import make_mapper
    from s2sbench.types import Word

    m1 = make_mapper("M1", lexicon)
    for txt in ["it's", "I'm", "I'd", "John's", "let's"]:
        labels = [i.label for i in m1.map([Word(txt, 0.0, 0.3)], batch_by_emit=False)]
        assert not any(lab in ("s", "m", "d") for lab in labels), (txt, labels)
