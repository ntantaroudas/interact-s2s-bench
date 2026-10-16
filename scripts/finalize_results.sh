#!/usr/bin/env bash
# After the streaming runs: derive P1v, score everything, make figures and build the manuscript.
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/derive_p1_vad.py --runs runs_cpu
python scripts/evaluate.py --runs runs_cpu --out results/streaming
if [ -d runs_cpu_repeat ]; then
  python scripts/repeatability.py --a runs_cpu --b runs_cpu_repeat --out results/streaming/repeatability.csv
fi
python scripts/spurious_breakdown.py --runs runs_cpu --out results/streaming
python scripts/make_figures_plain.py --example ES2004a 35 46.5
# the article sources are not part of every copy of the code
if [ -f manuscript/build_paper_v4.py ]; then
  python manuscript/build_paper_v4.py
  python manuscript/build_ore_docx.py
fi
