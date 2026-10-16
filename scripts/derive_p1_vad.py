#!/usr/bin/env python3
"""Derive "P1 with a VAD gate" (P1v) exactly from the P1 runs.

    python scripts/derive_p1_vad.py --runs runs_cpu --policy P1-h1.0-c0.25

P1 transcribes every window independently (no prompt, no state), so skipping the windows in which a
voice-activity detector finds no speech changes nothing else: the output of a gated P1 is the P1 output
of the windows that contain speech, with the same emission times.  This script runs the Silero VAD with
the settings used by P2 (threshold 0.5, 500 ms minimum silence, 100 ms padding) over each excerpt, keeps
the P1 windows (core +- context) that overlap a detected speech segment, and writes a run file for the
derived policy ``P1v-...`` next to the P1 runs.  Its compute time is the measured compute time of the
kept windows plus the measured VAD time.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2sbench.audio import SR, load_audio  # noqa: E402
from s2sbench.vad import fixed_vad_iterator  # noqa: E402

TICK = 0.04


def speech_segments(audio, threshold=0.5, min_silence_ms=500, speech_pad_ms=100):
    """Speech segments (s) found by the streaming Silero VAD fed with 40 ms packets (as in P2)."""
    vad = fixed_vad_iterator(threshold, min_silence_ms, speech_pad_ms)
    n = int(round(TICK * SR))
    segs, cur = [], None
    for i in range(0, len(audio), n):
        r = vad(audio[i:i + n])
        if not r:
            continue
        if "start" in r and "end" in r:
            s, e = r["start"] / SR, r["end"] / SR
            if cur is not None:
                s = min(s, cur)
            segs.append((s, e))
            cur = None
        elif "start" in r:
            cur = r["start"] / SR if cur is None else cur
        elif "end" in r and cur is not None:
            segs.append((cur, r["end"] / SR))
            cur = None
    if cur is not None:
        segs.append((cur, len(audio) / SR))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_cpu")
    ap.add_argument("--policy", default="P1-h1.0-c0.25")
    ap.add_argument("--audio-dir", default="data/ami/audio")
    a = ap.parse_args()
    for fp in sorted(glob.glob(os.path.join(a.runs, "*", a.policy + "__unaware", "*.json")) +
                     glob.glob(os.path.join(a.runs, "*", a.policy, "*.json"))):
        run = json.load(open(fp))
        h, c = float(run["policy_cfg"]["hop"]), float(run["policy_cfg"]["context"])
        start, length = run.get("start_seconds", 0.0) or 0.0, run.get("limit_seconds")
        wav = os.path.join(a.audio_dir, f"{run['meeting']}.Mix-Headset.wav")
        audio = load_audio(wav, start, None if length is None else start + length)
        t0 = time.time()
        segs = speech_segments(audio)
        vad_s = time.time() - t0

        def window_of(t_emit):
            # window k is processed when audio reaches (k+1)h + c, rounded up to the 40 ms tick grid
            return int(math.floor(t_emit - c - h + 1e-6))

        def has_speech(k):
            lo, hi = max(0.0, k * h - c), (k + 1) * h + c
            return any(s < hi and e > lo for s, e in segs)

        dur = run["duration"]
        last_k = int(math.ceil(dur / h)) - 1   # the final window is flushed at the end of the stream

        def keep(t_emit):
            k = last_k if t_emit >= dur - 1e-6 else min(window_of(t_emit), last_k)   # final flush at the end
            return has_speech(max(k, 0))

        words = [w for w in run["words"] if keep(w["emit"])]
        iters = [it for it in run.get("iterations", []) if it.get("c", 0.0) <= 0.05 or keep(it["t"])]
        dec = [it for it in run.get("iterations", []) if it.get("c", 0.0) > 0.05]
        kept = [it for it in dec if keep(it["t"])]
        pid = a.policy.replace("P1-", "P1v-", 1) + "-vad"
        out_dir = os.path.join(os.path.dirname(os.path.dirname(fp)), pid + ("__unaware" if not run["comp_aware"] else ""))
        os.makedirs(out_dir, exist_ok=True)
        rec = dict(run)
        rec.update({
            "policy_id": pid, "policy_cfg": {**run["policy_cfg"], "id": pid, "vad_gate": True},
            "processor": run["processor"] + "+vad-gate", "derived_from": os.path.relpath(fp),
            "words": words, "iterations": iters,
            "compute_total": sum(it["c"] for it in kept) + vad_s,
            "n_calls": len(kept), "vad_segments": [[round(s, 3), round(e, 3)] for s, e in segs],
            "vad_compute_s": vad_s,
        })
        rec["rtf"] = rec["compute_total"] / dur
        with open(os.path.join(out_dir, f"{run['meeting']}.json"), "w") as f:
            json.dump(rec, f)
        print(f"{run['meeting']}: kept {len(kept)}/{len(dec)} windows, {len(words)}/{len(run['words'])} words, "
              f"RTF {rec['rtf']:.3f} (P1 {run['compute_total'] / dur:.3f})")


if __name__ == "__main__":
    main()
