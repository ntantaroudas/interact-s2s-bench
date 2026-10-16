#!/usr/bin/env python3
"""Write the article into the official Open Research Europe Article Template.

    python manuscript/build_ore_docx.py            # INTERACT_StreamingS2S_ORE_v5.md -> INTERACT_StreamingS2S_ORE_v5.docx
    python manuscript/build_ore_docx.py --pdf      # also a PDF preview (needs LibreOffice)

build_paper_v4.py writes the Markdown with the sections in the order and with the labels of the template (Title, Authors
and affiliations, Abstract, Keywords, Plain language summary, main text, Data and software availability, Ethics and
consent, Competing interests, Grant information, Acknowledgements, Declaration of generative AI, References) and, as
Open Research Europe asks, with the figures and their legends and the tables with their captions and notes at the end,
in the sections Figure legends and Tables. This script turns that Markdown into blocks and writes them with
ore_template/fill_template.py into ore_template/ORE_Article_Template.docx, in the template's own styles (Cambria 9.5 pt
body text, Arial headings, Plain Table 1 tables), page set-up, header and footer. Table and figure numbers are Word SEQ
fields, and every "Table n" and "Figure n" in the text is a cross-reference (REF field with a hyperlink) to its caption.
Quotes and ellipses are set typographically (’ “ ” …), and "No.", "pp." and "et al." are followed by a non-breaking
space, as in the Word file made earlier with pandoc.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ore_template"))
from fill_template import fill  # noqa: E402

SRC = HERE / "INTERACT_StreamingS2S_ORE_v5.md"
OUT = HERE / "INTERACT_StreamingS2S_ORE_v5.docx"
AUTHORS = "Nikolaos D. Tantaroudas, Andrew J. McCracken, Ilias Karachalios"
ABSTRACT_HEADS = {"Background", "Methods", "Results", "Conclusions"}

LINK = r"<?(?:https?://[^\s<>]+|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})>?"
XREF = r"\b(?:Tables?|Figures?) \d+(?:(?:, | and | to |–)\d+)*"
TOKEN = re.compile(f"(?P<link>{LINK})|(?P<xref>{XREF})")


def plain_runs(text: str, base: dict, xrefs: bool) -> list[dict]:
    """Split plain text into runs: hyperlinks, cross-references to tables and figures, and text."""
    runs: list[dict] = []
    last = 0
    for m in TOKEN.finditer(text):
        if m.group("xref") and not xrefs:
            continue
        if m.start() > last:
            runs.append({**base, "t": text[last:m.start()]})
        if m.group("link"):
            link = m.group("link").strip("<>")
            trail = re.search(r"[.,;:)\]]+$", link)
            trailing = trail.group(0) if trail else ""
            if trailing:
                link = link[:-len(trailing)]
            url = link if link.startswith("http") else "mailto:" + link
            runs.append({**base, "t": link, "link": url})
            if trailing:
                runs.append({**base, "t": trailing})
        else:
            s = m.group("xref")
            kind = "Table" if s.startswith("Table") else "Figure"
            label = re.match(r"(Tables?|Figures?) ", s).group(0)
            runs.append({**base, "t": label})
            for part in re.split(r"(\d+)", s[len(label):]):
                if part.isdigit():
                    runs.append({**base, "xref": kind, "n": int(part)})
                elif part:
                    runs.append({**base, "t": part})
        last = m.end()
    if last < len(text):
        runs.append({**base, "t": text[last:]})
    return runs


def inline(text: str, xrefs: bool = True) -> list[dict]:
    """Runs of one line of the Markdown: **bold**, *italic*, ^superscript^, `code` (set as text), backslash escapes."""
    segs: list[tuple[str, dict]] = []
    buf, state = [], {"b": False, "i": False, "sup": False}
    i = 0

    def flush():
        if buf:
            segs.append(("".join(buf), {k: True for k, v in state.items() if v}))
            buf.clear()
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text) and not text[i + 1].isalnum():
            buf.append(text[i + 1])
            i += 2
            continue
        if text.startswith("**", i):
            flush()
            state["b"] = not state["b"]
            i += 2
            continue
        if c == "*":
            flush()
            state["i"] = not state["i"]
            i += 1
            continue
        if c == "^":
            flush()
            state["sup"] = not state["sup"]
            i += 1
            continue
        if c == "`":
            i += 1
            continue
        buf.append(c)
        i += 1
    flush()
    if any(state.values()):
        raise SystemExit(f"unbalanced markup in: {text[:120]}")
    runs: list[dict] = []
    for s, base in smarten(segs):
        runs += plain_runs(s, base, xrefs)
    return runs


def smarten(segs: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Typographic quotes and ellipses, as in the Word file made with pandoc: ' becomes ’ (the text has apostrophes
    only), " becomes “ after a space or an opening bracket and ” otherwise, and ... becomes …. Links are left as they
    are. The line is treated as a whole, so a quote next to bold or italic text is placed by the character before it."""
    full = "".join(s for s, _ in segs)
    protected = {k for m in re.finditer(LINK, full) for k in range(m.start(), m.end())}
    chars = list(full)
    for k, c in enumerate(chars):
        if k in protected:
            continue
        if c == "'":
            chars[k] = "’"
        elif c == '"':
            prev = chars[k - 1] if k else " "
            chars[k] = "\u201c" if prev.isspace() or prev in "([{\u2013\u2014/" else "\u201d"
    new = "".join(chars)
    # a non-breaking space after "No.", "pp." and "et al.", as pandoc writes it, so that they stay with what follows
    new = re.sub(r"(?<![A-Za-z])(No|pp|et al)\. (?=\S)", "\\1.\u00a0", new)
    out, pos = [], 0
    for s, base in segs:
        out.append((new[pos:pos + len(s)].replace("...", "…"), base))
        pos += len(s)
    return out


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse(md: str) -> list[dict]:
    lines = md.replace("\r\n", "\n").split("\n")
    blocks: list[dict] = []
    section = ""
    pending_img = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        m = re.match(r"^(#{1,2}) (.*)$", line)
        if m:
            text = m.group(2).strip()
            if len(m.group(1)) == 1:
                section = text
                if text != "Title":                      # fill_template writes the "Title" label itself
                    blocks.append({"type": "h1", "runs": inline(text, xrefs=False)})
            else:
                blocks.append({"type": "h2", "runs": inline(text, xrefs=False)})
        elif section == "Title":
            blocks.append({"type": "title", "runs": inline(line.strip(), xrefs=False)})
        elif section == "Abstract" and line.strip().strip("*") in ABSTRACT_HEADS and line.startswith("**"):
            blocks.append({"type": "h2", "runs": [{"t": line.strip().strip("*")}]})
        elif line.startswith("![]("):
            pending_img = re.match(r"^!\[\]\(([^)]+)\)", line).group(1)
        elif re.match(r"^\*\*Figure \d+\. ", line):
            n = int(re.match(r"^\*\*Figure (\d+)\. ", line).group(1))
            if not pending_img:
                raise SystemExit(f"Figure {n} has no image before its legend")
            legend = "**" + line[len(f"**Figure {n}. "):]
            blocks.append({"type": "figure", "n": n, "path": str(HERE / pending_img), "legend": inline(legend),
                           "alt": f"Figure {n}"})
            pending_img = None
        elif re.match(r"^\*\*Table \d+\.\*\* ", line):
            n = int(re.match(r"^\*\*Table (\d+)\.", line).group(1))
            caption = line[len(f"**Table {n}.** "):]
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            if len(rows) < 3 or not re.match(r"^\|[\s:|-]+\|$", rows[1].strip()):
                raise SystemExit(f"Table {n}: no Markdown table after its caption")
            widths = [max(1, c.count("-")) for c in split_row(rows[1])]
            notes = []
            k = i
            while k < len(lines) and not lines[k].strip():
                k += 1
            if k < len(lines) and lines[k].startswith("*Note:*"):
                notes.append(inline(lines[k].strip()))
                i = k + 1
            blocks.append({"type": "table", "n": n, "caption": inline(caption),
                           "header": [inline(c) for c in split_row(rows[0])],
                           "rows": [[inline(c) for c in split_row(r)] for r in rows[2:]],
                           "widths": widths, "notes": notes})
            continue
        elif line.startswith("|"):
            raise SystemExit(f"table without caption: {line[:80]}")
        elif line.startswith("- "):
            blocks.append({"type": "bullet", "runs": inline(line[2:].strip())})
        elif re.match(r"^\d+\. ", line):
            n, text = re.match(r"^(\d+)\. (.*)$", line).groups()
            if section == "References":
                blocks.append({"type": "ref", "n": n, "runs": inline(text.strip(), xrefs=False)})
            else:
                blocks.append({"type": "numbered", "runs": inline(text.strip())})
        else:
            blocks.append({"type": "p", "runs": inline(line.strip())})
        i += 1
    return blocks


def text_of(runs: list[dict]) -> str:
    return "".join(r.get("t", str(r.get("n", ""))) for r in runs)


def main(argv: list[str]) -> int:
    md = SRC.read_text(encoding="utf-8")
    blocks = parse(md)
    sections = [text_of(b["runs"]) for b in blocks if b["type"] == "h1"]
    print("sections:", " | ".join(["Title"] + sections))
    title = next(text_of(b["runs"]) for b in blocks if b["type"] == "title")
    info = fill(blocks, OUT, title, AUTHORS)
    mentions = sum(1 for b in blocks for r in (b.get("runs") or []) if r.get("xref"))
    print(f"wrote {OUT.name} ({OUT.stat().st_size:,} bytes): {info['tables']} tables, {info['figures']} figures, "
          f"{info['cross_references']} cross-references ({mentions} in the text, the others in captions and notes), "
          f"{info['hyperlinks']} hyperlinks, template parts unchanged: {len(info['unchanged_parts'])}")
    if "--pdf" in argv:
        subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(HERE / "build"), str(OUT)],
                       check=True, capture_output=True)
        print("preview:", HERE / "build" / (OUT.stem + ".pdf"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
