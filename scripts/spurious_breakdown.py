#!/usr/bin/env python3
"""Which items make up the spurious signing?  (descriptive companion to evaluate.py)

    python scripts/spurious_breakdown.py --runs runs_cpu --out results/streaming

For every run, the sign sequence produced from the hypothesis is aligned with the one produced from the
reference by the same mapper (as in the SER).  Hypothesis items that are insertions or substitutions are
counted by kind (lexical sign / fingerspelled string) and label, and each spurious item is classified by
the transcript error behind it:

* ``non-speech``  its source word lies more than 1 s from any reference word (output in silence/noise);
* ``fragment``    its source word is a proper prefix/suffix (>= 1 letter shorter) of a reference word of at
                  least three letters within 1 s, i.e. a word cut by a chunk boundary;
* ``other``       any other recognition or segmentation error.

Writes spurious_items_<mapper>.csv (policy, kind, label, count) and spurious_causes_<mapper>.csv
(policy, kind, cause, items per minute averaged over meetings).
"""
import argparse
import glob
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2sbench.io import load_reference, load_run  # noqa: E402
from s2sbench.lexicon import Lexicon  # noqa: E402
from s2sbench.mapper import make_mapper  # noqa: E402
from s2sbench.metrics import SpeechIndex, _align, map_reference, prepare_for_mapping  # noqa: E402
from s2sbench.normalize import strip_for_lookup  # noqa: E402


def classify(word, ref, sidx):
    if sidx.distance(word.mid) > 1.0:
        return "non-speech"
    h = strip_for_lookup(word.text)
    if h:
        for r in ref:
            if abs(r.mid - word.mid) <= 1.0:
                rt = strip_for_lookup(r.text)
                if len(rt) >= 3 and rt != h and len(h) < len(rt) and (rt.startswith(h) or rt.endswith(h)):
                    return "fragment"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_cpu")
    ap.add_argument("--ref", default="data/ami/ref")
    ap.add_argument("--lexicon", default="data/lexicon.json")
    ap.add_argument("--mapper", default="M0")
    ap.add_argument("--out", default="results/streaming")
    a = ap.parse_args()
    mp = make_mapper(a.mapper, Lexicon.from_json(a.lexicon))
    labels = defaultdict(Counter)
    causes = []
    for fp in sorted(glob.glob(os.path.join(a.runs, "*", "*", "*.json"))):
        run = load_run(fp)
        ref, _ = load_reference(os.path.join(a.ref, f"{run['meeting']}.json"), limit=run.get("limit_seconds"),
                                start=run.get("start_seconds", 0.0) or 0.0)
        sidx = SpeechIndex(ref)
        r_items = [i for i in map_reference(mp, ref) if i.played]
        hyp = prepare_for_mapping(run["words"])
        h_items = [i for i in mp.map(hyp, batch_by_emit=True) if i.played]
        al = _align([i.token for i in r_items], [i.token for i in h_items])
        cnt = Counter()
        if al is not None:
            for ch in al.alignments[0]:
                if ch.type in ("insert", "substitute"):
                    for j in range(ch.hyp_start_idx, ch.hyp_end_idx):
                        it = h_items[j]
                        labels[run["policy_id"]][(it.kind, it.label)] += 1
                        cnt[(it.kind, classify(hyp[it.src[-1]], ref, sidx))] += 1
        minutes = run["duration"] / 60.0
        for kind in ("LEX", "FS"):
            for cause in ("non-speech", "fragment", "other"):
                causes.append({"policy": run["policy_id"], "meeting": run["meeting"], "kind": kind, "cause": cause,
                               "per_min": cnt[(kind, cause)] / minutes})
    os.makedirs(a.out, exist_ok=True)
    rows = [{"policy": p, "kind": k, "label": lab, "count": n}
            for p, c in labels.items() for (k, lab), n in c.most_common()]
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"spurious_items_{a.mapper}.csv"), index=False)
    cz = pd.DataFrame(causes).groupby(["policy", "kind", "cause"], as_index=False)["per_min"].mean()
    cz.to_csv(os.path.join(a.out, f"spurious_causes_{a.mapper}.csv"), index=False)
    for p, c in labels.items():
        print(p, "top spurious:", ", ".join(f"{k}:{lab}×{n}" for (k, lab), n in c.most_common(12)))
    print(cz.pivot_table(index=["policy", "kind"], columns="cause", values="per_min").round(2))


if __name__ == "__main__":
    main()
