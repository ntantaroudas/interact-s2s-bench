#!/usr/bin/env python3
"""Run streaming policies on AMI meetings with the virtual-clock simulator.

    python scripts/run_grid.py --config configs/main.yaml
    python scripts/run_grid.py --config configs/smoke_cpu.yaml        # CPU smoke test

Each (model, policy, mode, meeting) produces runs/<model>/<policy_id>[__unaware]/<meeting>.json
holding the committed words with their emission times plus timing metadata.
Existing outputs are skipped, so the script can be stopped and restarted.
"""
import argparse
import gc
import json
import os
import platform
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from s2sbench.ami import PARTITIONS  # noqa: E402
from s2sbench.audio import load_audio  # noqa: E402
from s2sbench.policies import build_policy  # noqa: E402
from s2sbench.simulate import simulate  # noqa: E402


def env_info(device: str) -> dict:
    info = {"python": platform.python_version(), "platform": platform.platform()}
    for mod in ("faster_whisper", "ctranslate2", "torch", "numpy", "jiwer"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            pass
    try:
        import torch

        if device == "cuda" and torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    return info


def excerpt_bounds(wav: str, length, position: str = "middle", fixed_start: float = 0.0):
    """(start, length) of the excerpt to process; (0, None) = whole file.

    ``middle``: the excerpt is centred in the recording (start rounded to 0.01 s),
    ``start``: it begins at ``fixed_start`` seconds."""
    if not length:
        return 0.0, None
    import soundfile as sf

    dur = sf.info(wav).frames / sf.info(wav).samplerate
    length = float(min(length, dur))
    if position == "middle":
        start = round(max(0.0, (dur - length) / 2.0), 2)
    elif position == "start":
        start = float(fixed_start)
    else:
        raise ValueError(f"unknown excerpt_position {position}")
    return start, length


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--meetings", nargs="*")
    ap.add_argument("--models", nargs="*")
    ap.add_argument("--policies", nargs="*", help="subset of policy ids")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    meetings = a.meetings or cfg.get("meetings") or PARTITIONS[cfg.get("partition", "eval")]
    models = a.models or cfg["models"]
    policies = [p for p in cfg["policies"] if not a.policies or p["id"] in a.policies]
    device = cfg.get("device", "cuda")
    limit = cfg.get("limit_seconds")
    excerpt = cfg.get("excerpt_seconds")          # e.g. 180: one excerpt of this length per meeting
    excerpt_pos = cfg.get("excerpt_position", "middle")
    torch_threads = cfg.get("torch_threads")      # silero-vad sets torch to 1 thread on import
    modes = ([True] if cfg.get("comp_aware_runs", True) else []) + ([False] if cfg.get("comp_unaware") else [])
    unaware_models = set(cfg.get("comp_unaware_models", models))
    unaware_policies = set(cfg.get("comp_unaware_policies", [p["id"] for p in policies]))
    env = env_info(device)
    print(json.dumps(env, indent=1))

    for model in models:
        shared_asr = None
        for p in policies:
            for comp_aware in modes:
                if not comp_aware and (model not in unaware_models or p["id"] not in unaware_policies):
                    continue
                sub = p["id"] + ("" if comp_aware else "__unaware")
                out_dir = os.path.join(cfg.get("out_dir", "runs"), model, sub)
                todo = [m for m in meetings if not os.path.exists(os.path.join(out_dir, f"{m}.json"))]
                if not todo:
                    continue
                print(f"== {model} {sub}: {len(todo)} meeting(s)")
                if a.dry_run:
                    continue
                pcfg = dict(p)
                pcfg.update({"model": model, "device": device,
                             "compute_type": cfg.get("compute_type", "float16" if device == "cuda" else "int8"),
                             "download_root": cfg.get("download_root"),
                             "openai_model_dir": cfg.get("openai_model_dir", "models/openai")})
                if p["policy"] in ("fixed", "vad") and shared_asr is None:
                    from s2sbench.asr import FasterWhisperBackend

                    shared_asr = FasterWhisperBackend(model=model, device=device,
                                                      compute_type=pcfg["compute_type"],
                                                      beam_size=p.get("beam_size", 5),
                                                      download_root=pcfg["download_root"])
                proc = build_policy(pcfg, asr=shared_asr if p["policy"] in ("fixed", "vad") else None)
                os.makedirs(out_dir, exist_ok=True)
                for m in todo:
                    wav = os.path.join(cfg.get("audio_dir", "data/ami/audio"), f"{m}.Mix-Headset.wav")
                    start, seg_len = excerpt_bounds(wav, excerpt, excerpt_pos, cfg.get("excerpt_start", 0.0))
                    if seg_len is None:
                        seg_len = limit
                    audio = load_audio(wav, start, None if seg_len is None else start + seg_len)
                    if torch_threads:
                        import torch

                        torch.set_num_threads(int(torch_threads))
                    proc.warmup(audio)
                    proc.reset()
                    t0 = time.time()
                    res = simulate(proc, audio, comp_aware=comp_aware)
                    rec = {
                        "meeting": m, "model": model, "policy_id": p["id"], "policy_cfg": p,
                        "processor": getattr(proc, "name", p["policy"]), "comp_aware": comp_aware,
                        "device": device, "compute_type": pcfg["compute_type"], "limit_seconds": seg_len,
                        "start_seconds": start, "torch_threads": torch_threads,
                        "duration": res.duration, "compute_total": res.compute_total, "rtf": res.rtf,
                        "n_calls": res.n_calls, "stats": res.stats, "boundaries": res.boundaries,
                        "iterations": res.iterations, "env": env, "wallclock_s": time.time() - t0,
                        "words": [w.to_dict() for w in res.words],
                    }
                    tmp = os.path.join(out_dir, f"{m}.json.part")
                    with open(tmp, "w") as f:
                        json.dump(rec, f)
                    os.replace(tmp, os.path.join(out_dir, f"{m}.json"))
                    print(f"   {m}: {len(res.words)} words, RTF {res.rtf:.3f}, {time.time() - t0:.0f}s")
                del proc
                gc.collect()
        del shared_asr
        gc.collect()


if __name__ == "__main__":
    main()
