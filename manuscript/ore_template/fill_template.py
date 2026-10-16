"""Fill the official Open Research Europe Article Template with the manuscript.

    python manuscript/build_ore_docx.py      # calls fill() below

ORE_Article_Template.docx is the official Open Research Europe Article Template (Word, created 2022-12-14, last modified
2023-01-04, SHA-256 f5d96d7f2eca78e8e3f34889cb2be3d49964abbf8218293892d9fdf72790b64d), the same file as in the authors'
KyGround and ConstrainedTranslation studies, whose filler this module reuses with three changes: the cross-references
carry \\* MERGEFORMAT so that updating them in Word keeps them in plain type (the caption numbers they point to are
bold), the tables have the bold first column of the template's example table, and the text paragraphs have widow
and orphan control and each reference is kept on one page (the template switches widow control off in its document
defaults, which leaves single lines at the top or bottom of pages). The template is edited at the XML level. Its theme, settings, header
("https://open-research-europe.ec.europa.eu | editorial@..."), footer ("Information Classification: General",
"Page X of Y"), page set-up and custom properties are kept byte for byte. Otherwise

  * word/document.xml keeps the template's first paragraph ("Open Research Europe Article Template"), the empty
    paragraph after it and the section properties, and the instruction text in between is replaced by the manuscript,
    written with the template's own styles (Heading1 for section labels, Heading2 for subsections, BodyText for text,
    ListParagraph with the template's bullet list, PlainTable1 for tables);
  * word/styles.xml gains Word's built-in "caption" style, which the template lacks. Table and figure captions are
    Word captions, with the label in bold as in the template's examples ("Table 1.", "Figure 1."), the number as a
    SEQ field and a bookmark on it, so that Word renumbers them. A table caption stands above its table and a figure
    legend below its figure. Every "Table n" and "Figure n" in the text is a cross-reference (REF field) to that
    bookmark, so the references follow the numbering;
  * word/numbering.xml gains one decimal list (the template has bullets and a "[n]" list only);
  * word/_rels/document.xml.rels loses the relationships of the removed instruction text (hyperlinks to ORE's
    guidelines and the ORE logo used as a sample figure) and gains the manuscript's hyperlinks and figures;
  * docProps/core.xml carries the article's title and authors instead of the template's.

Blocks (from build_ore_docx.py) are dicts with a "type": title, h1, h2, p, bullet, numbered, ref, table, figure. Text
is a list of runs, dicts with "t" (text) and optionally b, i, sup, link (URL), hl (text the authors still have to
complete, highlighted) or xref ("Table" or "Figure", with "n", a cross-reference whose result is the number).
"""
from __future__ import annotations

import datetime as dt
import re
import struct
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "ORE_Article_Template.docx"
TEMPLATE_SHA256 = "f5d96d7f2eca78e8e3f34889cb2be3d49964abbf8218293892d9fdf72790b64d"

PAGE_W, MARGIN_L, MARGIN_R = 11910, 920, 920          # from the template's sectPr (A4)
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R               # 10070 DXA, 6.99 in
FIG_MAX_EMU = int(6.5 * 914400)                        # figures at the size they were drawn for, at most 6.5 in wide
BODY_SZ, TABLE_SZ, NOTE_SZ, TITLE_SZ, CAPTION_SZ = 19, 17, 17, 24, 18   # half-points (BodyText is 9.5 pt)
NEW_PAGE = {"Figure legends", "Tables"}
BULLET_NUMID, DECIMAL_NUMID = 3, 4                     # 3 = the template's bullet list; 4, 5, ... are added below

A = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC = "http://schemas.openxmlformats.org/drawingml/2006/picture"
REL_HYPERLINK = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink"
REL_IMAGE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"

CAPTION_STYLE = (
    '<w:style w:type="paragraph" w:styleId="Caption"><w:name w:val="caption"/><w:basedOn w:val="BodyText"/>'
    '<w:next w:val="BodyText"/><w:uiPriority w:val="35"/><w:unhideWhenUsed/><w:qFormat/>'
    '<w:pPr><w:spacing w:before="120" w:after="120"/><w:jc w:val="both"/></w:pPr>'
    f'<w:rPr><w:sz w:val="{CAPTION_SZ}"/><w:szCs w:val="{CAPTION_SZ}"/></w:rPr></w:style>')


def bookmark_name(kind: str, n: int) -> str:
    """Hidden bookmark on the number of a caption, named like Word's own (_Ref followed by digits)."""
    return f"_Ref{(1 if kind == 'Table' else 2)}{n:08d}"


class Doc:
    def __init__(self) -> None:
        self.rels: list[tuple[str, str, str, bool]] = []    # (id, type, target, external)
        self.links: dict[str, str] = {}
        self.media: list[tuple[str, bytes]] = []
        self.next_rid = 101
        self.next_pic = 101
        self.next_bm = 1
        self.decimal_lists = 0
        self.captions: dict[str, list[int]] = {"Table": [], "Figure": []}
        self.xrefs: list[tuple[str, int]] = []

    def rid(self) -> str:
        r = f"rId{self.next_rid}"
        self.next_rid += 1
        return r

    def link(self, url: str) -> str:
        if url not in self.links:
            r = self.rid()
            self.links[url] = r
            self.rels.append((r, REL_HYPERLINK, url, True))
        return self.links[url]

    def image(self, data: bytes, ext: str) -> str:
        n = len(self.media) + 1
        name = f"media/figure{n}.{ext}"
        self.media.append((f"word/{name}", data))
        r = self.rid()
        self.rels.append((r, REL_IMAGE, name, False))
        return r


# ------------------------------------------------------------------------------------------------ runs

def t_xml(text: str) -> str:
    parts = text.split("\t")
    out = []
    for k, p in enumerate(parts):
        if k:
            out.append("<w:tab/>")
        if p:
            out.append(f'<w:t xml:space="preserve">{escape(p)}</w:t>')
    return "".join(out)


def rpr(r: dict, size: int | None = None, bold: bool = False, link: bool = False, extra: str = "") -> str:
    x = []
    if link:
        x.append('<w:rStyle w:val="Hyperlink"/>')
    if r.get("b") or bold:
        x.append("<w:b/><w:bCs/>")
    if r.get("i"):
        x.append("<w:i/><w:iCs/>")
    if extra:
        x.append(extra)
    if size:
        x.append(f'<w:sz w:val="{size}"/><w:szCs w:val="{size}"/>')
    if r.get("hl"):                                    # text the authors still have to complete
        x.append('<w:highlight w:val="yellow"/>')
    if r.get("sup"):
        x.append('<w:vertAlign w:val="superscript"/>')
    return f"<w:rPr>{''.join(x)}</w:rPr>" if x else ""


def field(instr: str, result: str, props: str) -> str:
    """A complex field with its cached result."""
    return (f'<w:r>{props}<w:fldChar w:fldCharType="begin"/></w:r>'
            f'<w:r>{props}<w:instrText xml:space="preserve"> {escape(instr)} </w:instrText></w:r>'
            f'<w:r>{props}<w:fldChar w:fldCharType="separate"/></w:r>'
            f'<w:r>{props}<w:t>{escape(result)}</w:t></w:r>'
            f'<w:r>{props}<w:fldChar w:fldCharType="end"/></w:r>')


def runs_xml(doc: Doc, runs: list[dict], size: int | None = None, bold: bool = False) -> str:
    out = []
    for r in runs:
        if r.get("link"):
            rid = doc.link(r["link"])
            out.append(f'<w:hyperlink r:id="{rid}" w:history="1"><w:r>{rpr(r, size, bold, link=True)}'
                       f'{t_xml(r["t"])}</w:r></w:hyperlink>')
        elif r.get("xref"):
            doc.xrefs.append((r["xref"], r["n"]))
            out.append(field(f"REF {bookmark_name(r['xref'], r['n'])} \\h \\* MERGEFORMAT", str(r["n"]),
                             rpr(r, size, bold)))
        else:
            out.append(f"<w:r>{rpr(r, size, bold)}{t_xml(r['t'])}</w:r>")
    return "".join(out)


def para(doc: Doc, runs: list[dict], style: str = "BodyText", *, keep_next=False, keep_lines=False,
         page_break=False, num: tuple[int, int] | None = None, tabs: list[int] = (), before=None, after=None,
         line=None, ind: dict | None = None, jc: str | None = None, mark_sz: int | None = None,
         size: int | None = None, bold: bool = False, raw_runs: str = "") -> str:
    p = [f'<w:pStyle w:val="{style}"/>']
    if keep_next:
        p.append("<w:keepNext/>")
    if keep_lines:
        p.append("<w:keepLines/>")
    if page_break:
        p.append("<w:pageBreakBefore/>")
    p.append("<w:widowControl/>")                       # no single lines at the top or bottom of a page
    if num:
        p.append(f'<w:numPr><w:ilvl w:val="{num[1]}"/><w:numId w:val="{num[0]}"/></w:numPr>')
    if tabs:
        p.append("<w:tabs>" + "".join(f'<w:tab w:val="left" w:pos="{t}"/>' for t in tabs) + "</w:tabs>")
    sp = []
    if before is not None:
        sp.append(f'w:before="{before}"')
    if after is not None:
        sp.append(f'w:after="{after}"')
    if line is not None:
        sp.append(f'w:line="{line}" w:lineRule="auto"')
    if sp:
        p.append(f"<w:spacing {' '.join(sp)}/>")
    if ind:
        p.append("<w:ind " + " ".join(f'w:{k}="{v}"' for k, v in ind.items()) + "/>")
    if jc:
        p.append(f'<w:jc w:val="{jc}"/>')
    if mark_sz:
        p.append(f'<w:rPr><w:sz w:val="{mark_sz}"/><w:szCs w:val="{mark_sz}"/></w:rPr>')
    return f"<w:p><w:pPr>{''.join(p)}</w:pPr>{raw_runs}{runs_xml(doc, runs, size, bold)}</w:p>"


def text_of(runs: list[dict]) -> str:
    return "".join(r["t"] if "t" in r else str(r.get("n", "")) for r in runs)


def caption_xml(doc: Doc, kind: str, n: int, runs: list[dict], *, keep_next: bool, before: int, after: int) -> str:
    """'Table n.' or 'Figure n.' in bold, the number a SEQ field with a bookmark on it, then the caption text."""
    doc.captions[kind].append(n)
    bm_id = doc.next_bm
    doc.next_bm += 1
    b = "<w:rPr><w:b/><w:bCs/></w:rPr>"
    label = (f'<w:r>{b}<w:t xml:space="preserve">{kind} </w:t></w:r>'
             f'<w:bookmarkStart w:id="{bm_id}" w:name="{bookmark_name(kind, n)}"/>'
             + field(f"SEQ {kind} \\* ARABIC", str(n), b)
             + f'<w:bookmarkEnd w:id="{bm_id}"/><w:r>{b}<w:t xml:space="preserve">. </w:t></w:r>')
    return para(doc, runs, "Caption", keep_next=keep_next, keep_lines=True, before=before, after=after, jc="both",
                raw_runs=label)


# ------------------------------------------------------------------------------------------------ tables, figures

def scale_widths(widths: list[int] | None, n: int) -> list[int]:
    w = widths if widths and len(widths) == n else [1] * n
    tot = sum(w)
    out = [int(x * CONTENT_W / tot) for x in w]
    out[-1] += CONTENT_W - sum(out)
    return out


def table_xml(doc: Doc, b: dict) -> list[str]:
    n = len(b["header"])
    widths = scale_widths(b.get("widths"), n)
    out = [caption_xml(doc, "Table", b["n"], b["caption"], keep_next=True, before=240, after=100)]
    tbl = ['<w:tbl><w:tblPr><w:tblStyle w:val="PlainTable1"/>'
           f'<w:tblW w:w="{CONTENT_W}" w:type="dxa"/><w:tblLayout w:type="fixed"/>'
           '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" '
           'w:noHBand="0" w:noVBand="1"/></w:tblPr><w:tblGrid>'
           + "".join(f'<w:gridCol w:w="{w}"/>' for w in widths) + "</w:tblGrid>"]

    def cell(runs: list[dict], w: int, header: bool, keep: bool) -> str:
        p = para(doc, runs, keep_next=keep, keep_lines=True, before=20, after=20, line=240, size=TABLE_SZ,
                 bold=header, mark_sz=TABLE_SZ)
        return f'<w:tc><w:tcPr><w:tcW w:w="{w}" w:type="dxa"/></w:tcPr>{p}</w:tc>'

    tbl.append('<w:tr><w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>'
               + "".join(cell(c, widths[i], True, True) for i, c in enumerate(b["header"])) + "</w:tr>")
    notes = b.get("notes") or []
    for k, row in enumerate(b["rows"]):
        if len(row) != n:
            raise ValueError(f"Table {b['n']}: a row has {len(row)} cells, expected {n}")
        keep = k < len(b["rows"]) - 1 or bool(notes)    # the caption, the table and its notes stay together
        tbl.append("<w:tr><w:trPr><w:cantSplit/></w:trPr>"
                   + "".join(cell(c, widths[i], False, keep) for i, c in enumerate(row)) + "</w:tr>")
    tbl.append("</w:tbl>")
    out.append("".join(tbl))
    if not notes:                                       # space after the table
        out.append(para(doc, [], before=0, after=200, mark_sz=8))
    for k, note in enumerate(notes):
        last = k == len(notes) - 1
        out.append(para(doc, note, keep_next=not last, keep_lines=True, before=80 if k == 0 else 0,
                        after=360 if last else 40, size=NOTE_SZ, mark_sz=NOTE_SZ, jc="both"))
    return out


def image_info(data: bytes) -> tuple[str, int, int, float | None]:
    """Format, width and height in pixels, and resolution in dots per inch (None if the file gives none)."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        w, h = struct.unpack(">II", data[16:24])
        k = data.find(b"pHYs", 0, data.find(b"IDAT"))
        if k > 0:
            ppu, _, unit = struct.unpack(">IIB", data[k + 4:k + 13])
            if unit == 1:                               # pixels per metre
                return "png", w, h, ppu * 0.0254
        return "png", w, h, None
    raise ValueError("figures must be PNG")


def figure_xml(doc: Doc, b: dict) -> list[str]:
    data = Path(b["path"]).read_bytes()
    ext, w, h, dpi = image_info(data)
    cx = min(FIG_MAX_EMU, round(w / dpi * 914400)) if dpi else FIG_MAX_EMU
    cy = int(cx * h / w)
    rid = doc.image(data, ext)
    pid = doc.next_pic
    doc.next_pic += 1
    descr = escape(b.get("alt") or text_of(b["legend"])[:400], {'"': "&quot;"})
    drawing = (
        f'<w:r><w:rPr><w:noProof/></w:rPr><w:drawing><wp:inline distT="0" distB="0" distL="0" distR="0">'
        f'<wp:extent cx="{cx}" cy="{cy}"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{pid}" name="Figure {b["n"]}" descr="{descr}"/>'
        f'<wp:cNvGraphicFramePr><a:graphicFrameLocks xmlns:a="{A}" noChangeAspect="1"/></wp:cNvGraphicFramePr>'
        f'<a:graphic xmlns:a="{A}"><a:graphicData uri="{PIC}"><pic:pic xmlns:pic="{PIC}">'
        f'<pic:nvPicPr><pic:cNvPr id="{pid}" name="{escape(Path(b["path"]).name)}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        f'<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic>'
        f'</wp:inline></w:drawing></w:r>')
    return [para(doc, [], keep_next=True, before=240, after=0, jc="center", raw_runs=drawing),
            caption_xml(doc, "Figure", b["n"], b["legend"], keep_next=False, before=160, after=360)]


# ------------------------------------------------------------------------------------------------ body

def body_xml(doc: Doc, blocks: list[dict]) -> str:
    out: list[str] = []
    section = ""
    prev = {"type": "start"}
    for k, b in enumerate(blocks):
        nxt = blocks[k + 1] if k + 1 < len(blocks) else {"type": "end"}
        t = b["type"]
        if t == "title":
            out.append(para(doc, [{"t": "Title"}], "Heading1", keep_next=True, before=120))
            out.append(para(doc, b["runs"], size=TITLE_SZ, bold=True, after=120, line=264))
        elif t == "h1":
            section = text_of(b["runs"])
            out.append(para(doc, b["runs"], "Heading1", keep_next=True, page_break=section in NEW_PAGE,
                            before=0 if section in NEW_PAGE else 240))
        elif t == "h2":
            out.append(para(doc, b["runs"], "Heading2", keep_next=True, before=200, after=60, ind={"left": 0}))
        elif t == "p":
            if section == "Authors and affiliations":
                is_aff = bool(b["runs"]) and b["runs"][0].get("sup") and b["runs"][0]["t"][:1].isdigit()
                nxt_aff = nxt["type"] == "p" and nxt.get("runs") and nxt["runs"][0].get("sup") \
                    and nxt["runs"][0]["t"][:1].isdigit()
                out.append(para(doc, b["runs"], after=0 if (is_aff and nxt_aff) else 160))
            elif section == "Keywords":
                out.append(para(doc, b["runs"], after=120))
            else:
                out.append(para(doc, b["runs"], after=160, line=256, jc="both"))
        elif t in ("bullet", "numbered"):
            last = nxt["type"] != t
            if t == "numbered" and prev["type"] != "numbered":
                doc.decimal_lists += 1
            numid = BULLET_NUMID if t == "bullet" else DECIMAL_NUMID + doc.decimal_lists - 1
            out.append(para(doc, b["runs"], "ListParagraph", num=(numid, 0), tabs=[599], before=60,
                            after=160 if last else 0, line=256, size=BODY_SZ, mark_sz=BODY_SZ))
        elif t == "ref":
            out.append(para(doc, [{"t": f"{b['n']}.\t"}] + b["runs"], keep_lines=True, tabs=[425], after=60,
                            ind={"left": 425, "hanging": 425}))
        elif t == "table":
            out += table_xml(doc, b)
        elif t == "figure":
            out += figure_xml(doc, b)
        else:
            raise ValueError(f"unknown block {t}")
        prev = b
    return "".join(out)


# ------------------------------------------------------------------------------------------------ package

DECIMAL_ABSTRACT = (
    '<w:abstractNum w:abstractNumId="3"><w:multiLevelType w:val="hybridMultilevel"/>'
    '<w:lvl w:ilvl="0"><w:start w:val="1"/><w:numFmt w:val="decimal"/><w:lvlText w:val="%1."/>'
    '<w:lvlJc w:val="left"/><w:pPr><w:ind w:left="598" w:hanging="313"/></w:pPr>'
    '<w:rPr><w:rFonts w:ascii="Cambria" w:eastAsia="Cambria" w:hAnsi="Cambria" w:cs="Cambria" w:hint="default"/>'
    '<w:sz w:val="19"/><w:szCs w:val="19"/></w:rPr></w:lvl></w:abstractNum>')


def fill(blocks: list[dict], out_path: Path, title: str, authors: str) -> dict:
    import hashlib
    raw = TEMPLATE.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TEMPLATE_SHA256:
        raise SystemExit(f"{TEMPLATE} is not the official ORE Article Template")
    zin = zipfile.ZipFile(TEMPLATE)
    parts = {i.filename: zin.read(i.filename) for i in zin.infolist()}
    order = [i.filename for i in zin.infolist()]

    # document.xml: template prefix, the first two paragraphs, the manuscript, the template's sectPr
    d = parts["word/document.xml"].decode("utf-8")
    i0 = d.index("<w:body>") + len("<w:body>")
    j0 = d.rindex("<w:sectPr")
    paras = re.findall(r"<w:p[ >].*?</w:p>", d[i0:j0], re.S)
    head = paras[0] + paras[1]
    assert "Article Template" in re.sub(r"<[^>]+>", "", paras[0]), "unexpected template"
    doc = Doc()
    body = body_xml(doc, blocks)
    for kind, nums in doc.captions.items():             # captions numbered 1, 2, ... in order
        if nums != list(range(1, len(nums) + 1)):
            raise SystemExit(f"{kind} captions out of order: {nums}")
    missing = sorted({(k, n) for k, n in doc.xrefs if n not in doc.captions[k]})
    if missing:
        raise SystemExit(f"cross-references without a caption: {missing}")
    parts["word/document.xml"] = (d[:i0] + head + body + d[j0:]).encode("utf-8")

    # styles: Word's built-in caption style
    sx = parts["word/styles.xml"].decode("utf-8")
    assert 'w:styleId="Caption"' not in sx
    parts["word/styles.xml"] = sx.replace("</w:styles>", CAPTION_STYLE + "</w:styles>").encode("utf-8")

    # numbering: a decimal list (abstractNum 3) and one instance per numbered list, the later ones restarting at 1
    nx = parts["word/numbering.xml"].decode("utf-8")
    n_lists = max(1, doc.decimal_lists)
    assert 'w:abstractNumId="3"' not in nx
    k = nx.index("<w:num ")
    nx = nx[:k] + DECIMAL_ABSTRACT + nx[k:]
    nums = "".join(f'<w:num w:numId="{DECIMAL_NUMID + i}"><w:abstractNumId w:val="3"/>'
                   + ('<w:lvlOverride w:ilvl="0"><w:startOverride w:val="1"/></w:lvlOverride>' if i else "")
                   + "</w:num>" for i in range(n_lists))
    nx = nx.replace("</w:numbering>", nums + "</w:numbering>")
    parts["word/numbering.xml"] = nx.encode("utf-8")

    # relationships: drop those of the removed instruction text, add hyperlinks and figures
    rels = parts["word/_rels/document.xml.rels"].decode("utf-8")
    used = set(re.findall(r'r:(?:id|embed)="(rId\d+)"', parts["word/document.xml"].decode("utf-8")))
    kept, dropped = [], []
    for m in re.finditer(r"<Relationship [^>]*/>", rels):
        rel = m.group(0)
        rid = re.search(r'Id="([^"]+)"', rel).group(1)
        typ = re.search(r'Type="([^"]+)"', rel).group(1)
        if typ in (REL_HYPERLINK, REL_IMAGE) and rid not in used:
            dropped.append((rid, re.search(r'Target="([^"]+)"', rel).group(1)))
        else:
            kept.append(rel)
    for rid, typ, target, ext in doc.rels:
        kept.append(f'<Relationship Id="{rid}" Type="{typ}" Target="{escape(target, {chr(34): "&quot;"})}"'
                    + (' TargetMode="External"' if ext else "") + "/>")
    head_rels = rels[:rels.index("<Relationship ")]
    parts["word/_rels/document.xml.rels"] = (head_rels + "".join(kept) + "</Relationships>").encode("utf-8")
    for rid, target in dropped:
        if not target.startswith("http") and f"word/{target}" in parts:
            del parts[f"word/{target}"]
            order.remove(f"word/{target}")
    for name, data in doc.media:
        parts[name] = data
        order.append(name)
    ct = parts["[Content_Types].xml"].decode("utf-8")
    known = {e.lower() for e in re.findall(r'<Default Extension="([^"]+)"', ct)}   # compared ignoring case (OPC)
    for ext, mime in (("png", "image/png"), ("jpg", "image/jpeg")):
        if any(n.endswith("." + ext) for n, _ in doc.media) and ext not in known:
            ct = ct.replace("<Default ", f'<Default Extension="{ext}" ContentType="{mime}"/><Default ', 1)
    parts["[Content_Types].xml"] = ct.encode("utf-8")

    # core properties: the article instead of the template
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    core = parts["docProps/core.xml"].decode("utf-8")
    core = re.sub(r"<dc:creator>.*?</dc:creator>", f"<dc:creator>{escape(authors)}</dc:creator>", core)
    core = re.sub(r"<cp:lastModifiedBy>.*?</cp:lastModifiedBy>",
                  f"<cp:lastModifiedBy>{escape(authors.split(',')[0])}</cp:lastModifiedBy>", core)
    core = re.sub(r"<cp:revision>.*?</cp:revision>", "<cp:revision>1</cp:revision>", core)
    core = re.sub(r'(<dcterms:created xsi:type="dcterms:W3CDTF">).*?(</dcterms:created>)', rf"\g<1>{now}\g<2>", core)
    core = re.sub(r'(<dcterms:modified xsi:type="dcterms:W3CDTF">).*?(</dcterms:modified>)', rf"\g<1>{now}\g<2>", core)
    if "<dc:title>" in core:
        core = re.sub(r"<dc:title>.*?</dc:title>", f"<dc:title>{escape(title)}</dc:title>", core)
    else:
        core = core.replace("<dc:creator>", f"<dc:title>{escape(title)}</dc:title><dc:creator>", 1)
    parts["docProps/core.xml"] = core.encode("utf-8")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().timetuple()[:6]
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for name in order:
            info = zipfile.ZipInfo(name, date_time=stamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            zout.writestr(info, parts[name])
    return {"hyperlinks": len(doc.links), "figures": len(doc.media), "tables": len(doc.captions["Table"]),
            "cross_references": len(doc.xrefs), "dropped_relationships": dropped,
            "unchanged_parts": sorted(n for n in order if n in zin.namelist() and parts[n] == zin.read(n))}
