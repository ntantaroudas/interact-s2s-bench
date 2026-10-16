#!/usr/bin/env python3
"""Download the AMI material used by the benchmark and build reference files.

* manual annotations ami_public_manual_1.6.2.zip (word alignments)
* Mix-Headset audio (16 kHz mono) of the chosen partition
* data/ami/ref/<meeting>.json : time-ordered reference words of all speakers

AMI is distributed under CC BY 4.0 -- cite Carletta et al. (2005) and keep the
licence notice when redistributing derived files.

    python scripts/prepare_ami.py --partition eval --data data/ami
    python scripts/prepare_ami.py --partition eval --meetings ES2004a --data data/ami   # one meeting
"""
import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from s2sbench.ami import ANNOTATIONS_URL, AUDIO_URL, PARTITIONS, load_meeting_words, reference_segments  # noqa: E402


def download(url: str, dst: str) -> None:
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return
    tmp = dst + ".part"
    print(f"downloading {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    os.replace(tmp, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="eval", choices=sorted(PARTITIONS))
    ap.add_argument("--meetings", nargs="*", default=None, help="subset of meeting IDs")
    ap.add_argument("--data", default="data/ami")
    ap.add_argument("--no-audio", action="store_true", help="annotations/reference only")
    a = ap.parse_args()

    meetings = a.meetings or PARTITIONS[a.partition]
    man_zip = os.path.join(a.data, "ami_public_manual_1.6.2.zip")
    man_dir = os.path.join(a.data, "ami_public_manual_1.6.2")
    os.makedirs(a.data, exist_ok=True)
    if not os.path.isdir(os.path.join(man_dir, "words")):
        download(ANNOTATIONS_URL, man_zip)
        with zipfile.ZipFile(man_zip) as z:
            z.extractall(man_dir)

    ref_dir = os.path.join(a.data, "ref")
    audio_dir = os.path.join(a.data, "audio")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    summary = []
    for m in meetings:
        words = load_meeting_words(man_dir, m)
        with open(os.path.join(ref_dir, f"{m}.json"), "w", encoding="utf-8") as f:
            json.dump({"meeting": m, "source": "AMI manual annotations 1.6.2 (CC BY 4.0)",
                       "order": "segment", "words": [w.to_dict() for w in words],
                       "segments": reference_segments(words)}, f)
        if not a.no_audio:
            download(AUDIO_URL.format(m=m), os.path.join(audio_dir, f"{m}.Mix-Headset.wav"))
        last = max(w.end for w in words)
        summary.append((m, len(words), round(last / 60, 1)))
        print(f"{m}: {len(words)} words, last word at {last / 60:.1f} min")
    print(f"{len(summary)} meetings, {sum(s[1] for s in summary)} words")


if __name__ == "__main__":
    main()
