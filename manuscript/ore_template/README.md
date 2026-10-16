# Open Research Europe Article Template

`ORE_Article_Template.docx` is the official Open Research Europe Article Template (Word), as provided by Open
Research Europe (https://open-research-europe.ec.europa.eu) to its authors. It is included unchanged
(SHA-256 f5d96d7f2eca78e8e3f34889cb2be3d49964abbf8218293892d9fdf72790b64d) so that the Word file of the article
can be rebuilt, and it is not covered by the MIT License of this repository.

`fill_template.py` writes the article into a copy of the template with the template's own styles, header,
footer and page set-up, the figure legends and tables at the end of the manuscript, numbered captions and
cross-references. It is called by `manuscript/build_ore_docx.py`:

```
python manuscript/build_paper_v4.py          # -> manuscript/INTERACT_StreamingS2S_ORE_v5.md
python manuscript/build_ore_docx.py --pdf    # -> manuscript/INTERACT_StreamingS2S_ORE_v5.docx (+ PDF preview)
```
