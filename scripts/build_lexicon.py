#!/usr/bin/env python3
"""Build data/lexicon.json (keys, frame counts, hand-visible spans) from the
INTERACT landmark files (https://github.com/ntantaroudas/isl-extractions-main,
``landmarks/``; archived at https://doi.org/10.5281/zenodo.18656296).

Only metadata is written -- no landmark coordinates.

    python scripts/build_lexicon.py --landmarks ../isl-extractions-main/landmarks --out data/lexicon.json
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2sbench.lexicon import Lexicon  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", required=True)
    ap.add_argument("--out", default="data/lexicon.json")
    ap.add_argument("--source", default="https://github.com/ntantaroudas/isl-extractions-main (landmarks/)")
    a = ap.parse_args()
    lx = Lexicon.from_landmarks(a.landmarks)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    lx.to_json(a.out, source=a.source)
    print(json.dumps(lx.stats(), indent=1))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
