#!/usr/bin/env bash
# Clone the reference implementations of P3 (LocalAgreement-2) and P4 (AlignAtt)
# at the exact commits used in the paper.  whisper_streaming and SimulStreaming are MIT-licensed;
# simul_whisper (used only for its CIF end-of-word checkpoints) is Apache-2.0.
set -euo pipefail
DEST="${S2S_THIRD_PARTY:-$(cd "$(dirname "$0")/.." && pwd)/third_party}"
mkdir -p "$DEST"
fetch () {  # repo dir commit
  if [ ! -d "$DEST/$2/.git" ]; then git clone -q "https://github.com/$1" "$DEST/$2"; fi
  git -C "$DEST/$2" fetch -q --depth 1 origin "$3" 2>/dev/null || git -C "$DEST/$2" fetch -q origin
  git -C "$DEST/$2" checkout -q "$3"
  echo "$1 @ $(git -C "$DEST/$2" rev-parse --short HEAD)"
}
fetch ufal/whisper_streaming     whisper_streaming 6da90b44b7e50d79695e68166d2a2c7609c75abb
fetch ufal/SimulStreaming        SimulStreaming    077ea37d5ab4ff98bc567e4507f140dc4e5d5ad6
fetch backspacetg/simul_whisper  simul_whisper     ffeb3ff333026f29053d01c95a1a0524f2a41865
echo "third-party code in $DEST"
