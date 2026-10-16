#!/usr/bin/env bash
# CPU study of the paper: runs configs/cpu_small.yaml meeting by meeting (series interleaved, so
# partial results stay balanced), then the repeatability runs (configs/cpu_repeat.yaml).
# Restartable: finished (model, policy, meeting) files are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
export S2S_THIRD_PARTY=${S2S_THIRD_PARTY:-$PWD/third_party}
for m in ES2004a IS1009a TS3003a EN2002a ES2004b IS1009b TS3003b EN2002b \
         ES2004c IS1009c TS3003c EN2002c ES2004d IS1009d TS3003d EN2002d; do
  while [ ! -f "data/ami/audio/$m.Mix-Headset.wav" ]; do sleep 30; done   # audio still downloading
  python scripts/run_grid.py --config configs/cpu_small.yaml --meetings "$m"
done
python scripts/derive_p1_vad.py --runs runs_cpu       # P1v: P1 with a VAD gate, derived from the P1 runs
python scripts/run_grid.py --config configs/cpu_repeat.yaml
echo "CPU study finished"
