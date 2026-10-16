# INTERACT streaming speech-to-sign benchmark (`s2sbench`)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23410736.svg)](https://doi.org/10.5281/zenodo.23410736)

Measures how **live (streaming) transcription settings affect a real-time signing avatar**.
Meeting audio is replayed in real time through streaming ASR policies, the committed words are
mapped to the INTERACT International Sign (IS) lexicon, and a sequential avatar model plays the
clips. The benchmark scores the transcript, the sign stream and the avatar timing, and it includes
an *oracle* analysis that uses the manual transcript in place of the ASR.

It accompanies the article *How live transcription shapes a signing avatar: an open benchmark of
streaming speech recognition for real-time International Sign translation in meetings* by
N. D. Tantaroudas, A. J. McCracken and I. Karachalios, submitted to Open Research Europe, and its
shorter preprint on Preprints.org. The benchmark was built for INTERACT, an extended reality meeting
platform developed under the CORTEX2 Horizon Europe programme.

## Related outputs

| Output | Link |
|---|---|
| Preprint, *Does a Faster Transcript Make a Better Signing Avatar? Streaming Speech Recognition for a Live International Sign Avatar* | https://doi.org/10.20944/preprints202610.0572.v1 |
| International Sign dictionary used by the benchmark (landmark data of INTERACT) | https://doi.org/10.5281/zenodo.18656296 |
| Archived release of this software and data, version 1.0.0 | https://doi.org/10.5281/zenodo.23410736 |
| All versions of this software | https://doi.org/10.5281/zenodo.23410735 |

```
Meeting audio ─► streaming ASR policy ─► committed words ─► text-to-sign mapper ─► avatar playback
                 P1 fixed chunks          (text, ASR times,  M0 deployed           FIFO or
                 P2 pause/VAD              emission time)    M1 load-reduced       bounded lag
                 P3 LocalAgreement-2                         M2 no fingerspelling
                 P4 AlignAtt
```

## Repository layout

| Path | Content |
|---|---|
| `s2sbench/ami.py` | AMI partitions, NXT word/segment parsing, segment-ordered references |
| `s2sbench/policies/` | P1 `fixed.py` and P2 `vad_segment.py` (written for this benchmark), P3 `local_agreement.py` and P4 `alignatt.py` (adapters around the unchanged reference implementations). P1v (P1 + VAD gate) is derived from the P1 runs by `scripts/derive_p1_vad.py` |
| `s2sbench/simulate.py` | virtual-clock, computation-aware real-time simulator |
| `s2sbench/lexicon.py`, `mapper.py`, `avatar.py` | IS lexicon (exact Redis key rule), mappers M0–M2, playback model |
| `s2sbench/metrics.py`, `stats.py` | transcript/sign/avatar metrics, bootstrap CIs, Friedman + Wilcoxon–Holm |
| `scripts/` | data preparation, oracle analysis, streaming runs (`run_grid.py`, `run_cpu_study.sh`), scoring, repeatability, figures |
| `configs/` | `cpu_small.yaml` + `cpu_repeat.yaml` (the paper's streaming experiment), `cpu_test.yaml` (quick check), `core.yaml` / `ablations.yaml` / `repeat.yaml` (optional GPU replication), `smoke_cpu.yaml` |
| `data/lexicon.json` | lexicon metadata (keys, frame counts, hand-visible spans), no landmark coordinates |
| `data/ami/eval_durations.json` | exact durations of the 16 evaluation meetings |
| `results/oracle/` | RQ1 results (oracle analysis, full meetings) |
| `results/streaming/` | RQ2–RQ4 results (streaming experiment): per-meeting and summary tables, statistics, repeatability, spurious-item breakdown, excerpt list |
| `runs_cpu/`, `runs_cpu_repeat/` | run files: committed words with ASR times and emission times, compute times |
| `manuscript/` | template, reference list and build script of the Open Research Europe article (`build_paper_v4.py` fills every table from `results/`), Word conversion, and the figures in `figures_v4/` (EPS, PDF, PNG and `figure_check.md`) |
| `tests/` | unit tests (policies with a mock recogniser, mapper, scheduler, metrics) |

## Reproducing the paper's results

Everything in the paper runs on an ordinary CPU machine, and no GPU is needed.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchaudio
pip install -r requirements.txt
bash scripts/fetch_third_party.sh                  # pinned Whisper-Streaming, SimulStreaming, Simul-Whisper
export S2S_THIRD_PARTY=$PWD/third_party
python -m pytest -q                                 # unit tests
python scripts/prepare_ami.py --partition eval --data data/ami        # 16 meetings, about 0.6 GB of audio

# RQ1 - oracle analysis on the full meetings (about 1 minute)
python scripts/oracle_analysis.py

# RQ2-RQ4 - streaming experiment: Whisper small, one 3-minute excerpt from the middle of each
# meeting, computation-unaware simulation (7.4 hours on 2 vCPUs including the repeat, restartable)
bash scripts/run_cpu_study.sh                       # configs/cpu_small.yaml + configs/cpu_repeat.yaml
python scripts/derive_p1_vad.py --runs runs_cpu     # P1v: P1 with a VAD gate, derived exactly from P1
python scripts/evaluate.py --runs runs_cpu --out results/streaming
python scripts/repeatability.py --a runs_cpu --b runs_cpu_repeat --out results/streaming/repeatability.csv
python scripts/spurious_breakdown.py --runs runs_cpu --out results/streaming

# figures and article: figures drawn to the figure guidelines of Open Research Europe (150 or 75 mm wide,
# 8 pt text, RGB, EPS and PDF, uncompressed 600 dpi TIFF, 300 dpi PNG, checked in figures_v4/figure_check.md)
python scripts/make_figures_plain.py --example ES2004a 35 46.5
python manuscript/build_paper_v4.py                 # -> manuscript/INTERACT_StreamingS2S_ORE_v4.md
python manuscript/make_docx.py "$PWD/manuscript/INTERACT_StreamingS2S_ORE_v4.md"   # Word file, needs pandoc
```

`bash scripts/finalize_results.sh` runs the scoring, figure and article steps after the streaming runs in
one go. The run files and results of the study are included, so the figures and every table of the article
can be rebuilt from them without running the recognisers again.

Computation-unaware runs do not depend on processor speed: the committed words and their emission times
do not change with the machine's speed, and only the measured compute time (RTF) depends on the hardware.
One source of randomness remains: when a window fails faster-whisper's quality thresholds it is decoded
again by random sampling at a higher temperature. In the repeat run of the study this changed 2 of 400
words for P1 and nothing for P2-P4 (`scripts/repeatability.py` compares two runs).

### Quick check (about 5 minutes)

`python scripts/run_grid.py --config configs/cpu_test.yaml` runs all four policies on a 20-second
excerpt of one meeting. Its numbers are a pipeline check, not results.

## Optional: GPU replication with Whisper large-v2

INTERACT deploys Whisper large-v2 on an NVIDIA T4 (AWS g4dn.xlarge). `configs/core.yaml` runs the four
policies on the full meetings with large-v2 and small, in computation-aware mode (compute time on the
GPU adds to the latency), plus the computation-unaware floor, and `configs/repeat.yaml` repeats four meetings
to measure run-to-run variability, which computation-aware runs have.

1. **Environment.** Ubuntu 22.04 with an NVIDIA driver for CUDA 12, for example an AWS Deep
   Learning AMI. faster-whisper/CTranslate2 needs cuBLAS and cuDNN 9 for CUDA 12. Either use an
   image that has them, or install them with pip (from the faster-whisper documentation):
   ```bash
   pip install nvidia-cublas-cu12 nvidia-cudnn-cu12==9.*
   export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`
   ```
   Install a CUDA build of PyTorch (https://pytorch.org/get-started/locally/), then
   `pip install -r requirements.txt`, `bash scripts/fetch_third_party.sh` and
   `export S2S_THIRD_PARTY=$PWD/third_party`.
2. **Runs.**
   ```bash
   python scripts/run_grid.py --config configs/core.yaml        # main GPU grid
   python scripts/run_grid.py --config configs/ablations.yaml   # descriptive variants
   python scripts/run_grid.py --config configs/repeat.yaml      # run-to-run variability
   python scripts/evaluate.py --runs runs --out results/streaming_gpu --usd-per-hour <on-demand price>
   python scripts/repeatability.py --a runs --b runs_repeat --out results/streaming_gpu/repeatability.csv
   ```
   Runs are restartable. **Run computation-aware jobs alone on the GPU**, because their compute times
   are measurements. As a planning bound, budget roughly 40–120 GPU-hours for `core.yaml`.

## Modelling assumptions (stated in the paper)

* **M0** reproduces the behaviour described for INTERACT: exact lower-cased word lookup, with
  fingerspelling as the fallback. The lookup key follows `populate-redis.py` exactly.
* **Clip timing** is at 30 fps, using the hand-visible span (default) or the full clip.
  Faster playback is the parameter `rate`.
* **Avatar playback** is sequential, with no interruption and no transition time. This is
  optimistic for the avatar.
* **P2** approximates the pause-based Mediation Gateway transcription with a Silero VAD, because the gateway
  itself is not public.
* **P4** runs on PyTorch Whisper (SimulStreaming's requirement), whereas P1–P3 use CTranslate2.
  Compare compute cost with this in mind.
* **Streaming experiment:** Whisper small on CPUs, computation-unaware (the latencies are those of the
  policies, and processing time on real hardware comes on top), one 3-minute excerpt from the middle of
  each meeting. The silero-vad package sets PyTorch to one thread when imported, and `torch_threads` in the
  config restores the thread count before each run.

## Licences and attribution

* Benchmark code: MIT (see `LICENSE`).
* Run files (`runs_cpu/`, `runs_cpu_repeat/`), results (`results/`), dictionary metadata
  (`data/lexicon.json`) and figures: CC BY 4.0 (see `LICENSE-DATA.md`). They contain automatic
  transcripts of AMI recordings, so please also cite the AMI corpus (below).
* Vendored code is listed in `s2sbench/third_party/PROVENANCE.md`: the Whisper normaliser (MIT) and
  the Silero VAD iterator (MIT).
* P3/P4 use ufal/whisper_streaming (MIT) and ufal/SimulStreaming (MIT) at pinned commits, and the
  Simul-Whisper CIF models (Apache-2.0).
* AMI Meeting Corpus: CC BY 4.0. Please cite Carletta et al. (2006),
  https://doi.org/10.1007/11677482_3.
* INTERACT IS lexicon: https://doi.org/10.5281/zenodo.18656296.

## Citation

Please cite the article (or, until it is published, the preprint) and the archived release of this
software. `CITATION.cff` gives the details, and GitHub shows them under "Cite this repository".

## Funding

This work was supported by FSTP funding from the Horizon Europe research and innovation programme of the
European Union under grant agreement No. 101070192 (CORTEX2). Views and opinions expressed are those of the
authors only and do not necessarily reflect those of the European Union or the granting authority.
