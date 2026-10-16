#!/usr/bin/env python3
"""Convert the manuscript Markdown to a Word file and tidy the tables.

    python manuscript/make_docx.py        # INTERACT_StreamingS2S_ORE_v3.md -> INTERACT_StreamingS2S_ORE_v3.docx

pandoc does the conversion (figures are embedded from manuscript/figures).  python-docx then
* sets the text of all tables to 9 pt (the relative column widths written by build_paper.py assume this),
* repeats each table's header row when a table runs onto a new page,
* keeps every table caption with its table, every table header with the first row, and every figure with
  its caption,
* sets an A4 page with 25 mm margins and embeds every figure at its printed size (pixels divided by dpi).
"""
import io
import os
import subprocess
import sys

from docx import Document
from docx.shared import Mm
from PIL import Image
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "INTERACT_StreamingS2S_ORE_v3.md")
TABLE_PT = 9


def _set(parent, tag, **attrs):
    el = parent.find(qn(tag))
    if el is None:
        el = OxmlElement(tag)
        parent.append(el)
    for k, v in attrs.items():
        el.set(qn(k), v)
    return el


def _keep_with_next(p):
    _set(p.get_or_add_pPr(), "w:keepNext")


def main(src=SRC):
    dst = os.path.splitext(src)[0] + ".docx"
    subprocess.run(["pandoc", os.path.basename(src), "-o", dst, "--resource-path=.:figures"],
                   cwd=os.path.dirname(src), check=True)
    doc = Document(dst)
    for t in doc.tables:
        tbl = t._tbl
        for r in tbl.iter(qn("w:r")):
            rpr = r.get_or_add_rPr()
            _set(rpr, "w:sz", **{"w:val": str(2 * TABLE_PT)})
            _set(rpr, "w:szCs", **{"w:val": str(2 * TABLE_PT)})
        rows = tbl.findall(qn("w:tr"))
        if rows:
            _set(rows[0].get_or_add_trPr(), "w:tblHeader")
            for p in rows[0].iter(qn("w:p")):
                _keep_with_next(p)
    # A4 page with 25 mm margins (160 mm of text width), and every figure at its printed size (pixels / dpi),
    # so that its 8 pt text stays 8 pt: pandoc reads the PNG resolution of 299.9994 dpi as 299 and scales.
    for sec in doc.sections:
        sec.page_width, sec.page_height = Mm(210), Mm(297)
        sec.left_margin = sec.right_margin = sec.top_margin = sec.bottom_margin = Mm(25)
    text_w = doc.sections[0].page_width - doc.sections[0].left_margin - doc.sections[0].right_margin
    for sh in doc.inline_shapes:
        blob = doc.part.related_parts[sh._inline.graphic.graphicData.pic.blipFill.blip.embed].blob
        im = Image.open(io.BytesIO(blob))
        dpi = round(float(im.info.get("dpi", (300, 300))[0])) or 300
        cx, cy = round(im.width * 914400 / dpi), round(im.height * 914400 / dpi)
        if cx <= text_w:
            sh.width, sh.height = cx, cy
    body = list(doc.element.body.iterchildren())
    n_cap = n_fig = 0
    for el, nxt in zip(body, body[1:]):
        if el.tag != qn("w:p"):
            continue
        text = "".join(x.text or "" for x in el.iter(qn("w:t"))).strip()
        if text.startswith("Table ") and nxt.tag == qn("w:tbl"):
            _keep_with_next(el)
            n_cap += 1
        elif next(el.iter(qn("w:drawing")), None) is not None:
            _keep_with_next(el)
            n_fig += 1
    doc.save(dst)
    print(f"wrote {dst}: {len(doc.tables)} tables ({n_cap} captions kept with their tables), {n_fig} figures")


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
