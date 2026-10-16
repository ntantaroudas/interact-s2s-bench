# CPU smoke test (pipeline check — not results)

Run on 5 October 2026 in a 2-core cloud VM without a GPU: Whisper **tiny** (int8), the **first 3 minutes**
of AMI meeting ES2004a, computation-aware simulation (`configs/smoke_cpu.yaml`).

Its only purpose is to show that every stage runs end to end — the four policies, the mappers, the
scheduler and the metrics. The numbers come from a tiny model, a CPU and 3 minutes of one meeting.
They must **not** be reported or interpreted as findings; the paper's streaming results come from the
CPU study in `configs/cpu_small.yaml` (Whisper small, computation-unaware; `scripts/run_cpu_study.sh`).

| Policy | WER | Median emission lag (s) | FRAG /1k | DUP /1k | Spurious FS/min (M0) | RTF (2-core CPU) |
|---|---|---|---|---|---|---|
| P1-h1.0-c0.25 | 0.390 | 1.08 | 3.5 | 24.5 | 29.3 | 0.50 |
| P1-h1.0-c0 | 0.549 | 0.87 | 3.5 | 14.0 | 38.3 | 0.49 |
| P2-sil500 | 0.292 | 3.09 | 10.5 | 0.0 | 15.7 | 0.09 |
| P3-la-1.0 | 0.260 | 2.42 | 7.0 | 0.0 | 10.3 | 0.36 |
| P4-aa-1.0 | 0.295 | 1.88 | 10.5 | 0.0 | 15.3 | 0.51 |

Checks performed
* All four policies (P1–P4) produced time-stamped commits; P3/P4 ran the unchanged reference code.
* Every metric was finite, and the sign alignment handled multi-word signs.
* Computation-unaware mode ran for P1, P3 and P4.
* A duplicated end-of-stream flush in the P3 adapter was found and fixed (see
  `s2sbench/policies/local_agreement.py`).
* P4 (SimulStreaming) occasionally repeats a word at the end of a voice segment. This is genuine
  behaviour of the reference implementation and is counted by the DUP metric.
* Two computation-aware P3 runs on the same minute differed (25 vs 41 words) because of timing
  jitter combined with the reference end-of-segment flush. This motivates `configs/repeat.yaml` and
  the `P3-la-1.0-fd` ablation.
