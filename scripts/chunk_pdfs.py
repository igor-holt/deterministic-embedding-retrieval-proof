#!/usr/bin/env python3
"""Chunk the 10 selected PDFs into fixed 500-char / 50-char-overlap chunks.

CHUNKING RULE (v1, exact — re-running this script on the same inputs must
produce byte-identical chunk_table.json):
  1. Text extraction: `pdftotext -enc UTF-8 <pdf> -` (poppler; no -layout,
     no -raw). One PDF per run; pdftotext version recorded in the output.
  2. Normalization: every maximal run of whitespace (space, tab, newline,
     form feed, etc.) is replaced by a single ASCII space; leading/trailing
     whitespace stripped. Result is a single line of text per PDF.
  3. Chunking over the normalized text, with sentence-boundary snapping:
     - Target chunk length T = 500 characters; overlap O = 50 characters.
     - Start s = 0. While s < len(text):
         e = min(s + T, len(text)).
         If e < len(text): look for the LAST sentence terminator in
         text[s+250 : e] — a terminator is one of '.', '!', '?' followed by
         a space or by end-of-text. If one is found at relative position p
         (absolute position s+250+p), set e = s+250+p+1 (the chunk ends just
         after the terminator). If none is found, keep e = s + T (hard cut).
         If e - s < 100 and e < len(text): extend e to min(s + T, len(text))
         (never emit a <100-char chunk except as the final chunk).
         Emit chunk text[s:e] with char_start=s, char_end=e.
         If e == len(text): stop. Else s = e - O.
     - If the final chunk is < 100 chars and there is a previous chunk,
       merge it into the previous chunk (extend the previous chunk's
       char_end to len(text) and drop the final chunk). (A PDF whose whole
       normalized text is < 100 chars yields a single chunk.)
  4. Chunk ids are global and sequential in corpus order:
     <pdf_index:02d>-<chunk_index:03d>, e.g. 00-000. Corpus order is the
     order of `selected` in corpus-selection.json.
  5. A PDF with empty normalized text contributes 0 chunks (recorded).

Outputs: chunk_table.json (full table) and chunk_table.csv (flat view).
"""
import json
import os
import subprocess
import sys

PROOF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEL_PATH = os.path.join(PROOF, "corpus-selection.json")
OUT_JSON = os.path.join(PROOF, "chunk_table.json")
OUT_CSV = os.path.join(PROOF, "chunk_table.csv")
WS = os.path.expanduser("~/workspace")
TARGET, OVERLAP, MIN_SNAP, MIN_CHUNK = 500, 50, 250, 100


def pdftotext_version():
    r = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True)
    return (r.stderr or r.stdout).strip().splitlines()[0] if (r.stderr or r.stdout) else "unknown"


def extract(pdf_path):
    r = subprocess.run(["pdftotext", "-enc", "UTF-8", pdf_path, "-"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit("pdftotext failed on %s: %s" % (pdf_path, r.stderr[:500]))
    return r.stdout


def normalize(raw):
    return " ".join(raw.split())


def chunk_text(text):
    """Return list of (char_start, char_end) per the rule in the docstring."""
    n = len(text)
    spans = []
    s = 0
    while s < n:
        e = min(s + TARGET, n)
        if e < n:
            window = text[s + MIN_SNAP:e]
            # last terminator followed by a space (or end of text)
            best = -1
            for i, ch in enumerate(window):
                if ch in ".!?" and (s + MIN_SNAP + i + 1 >= n or
                                    text[s + MIN_SNAP + i + 1] == " "):
                    best = i
            if best >= 0:
                e = s + MIN_SNAP + best + 1
            if e - s < MIN_CHUNK:
                e = min(s + TARGET, n)
        spans.append((s, e))
        if e >= n:
            break
        s = e - OVERLAP
        if s <= spans[-1][0]:
            s = spans[-1][0] + 1  # safety: never stall
    # merge a short final chunk into the previous one
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < MIN_CHUNK:
        spans[-2] = (spans[-2][0], spans[-1][1])
        spans.pop()
    return spans


def main():
    sel = json.load(open(SEL_PATH))
    tool = pdftotext_version()
    rows = []
    per_pdf = []
    for pi, pdf in enumerate(sel["selected"]):
        path = os.path.join(WS, pdf["path"].lstrip("./"))
        raw = extract(path)
        text = normalize(raw)
        spans = chunk_text(text)
        per_pdf.append({"pdf_index": pi, "path": pdf["path"], "sha256": pdf["sha256"],
                        "size_bytes": pdf["size"], "text_chars": len(text),
                        "chunk_count": len(spans)})
        for ci, (a, b) in enumerate(spans):
            rows.append({"chunk_id": "%02d-%03d" % (pi, ci), "pdf_index": pi,
                         "chunk_index": ci, "char_start": a, "char_end": b,
                         "text": text[a:b]})
    out = {"rule": "chunking v1: pdftotext -enc UTF-8, whitespace collapsed to single "
                   "spaces, target 500 chars, 50-char overlap, sentence-boundary snap "
                   "(last . ! ? + space in [start+250, start+500)), min chunk 100 chars "
                   "(short final chunk merged into previous)",
            "tool": tool, "seed": sel["seed"],
            "total_chunks": len(rows), "per_pdf": per_pdf, "chunks": rows}
    json.dump(out, open(OUT_JSON, "w"), indent=2, ensure_ascii=False)
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("chunk_id,pdf_index,chunk_index,char_start,char_end,text_chars\n")
        for r in rows:
            f.write("%s,%d,%d,%d,%d,%d\n" % (r["chunk_id"], r["pdf_index"], r["chunk_index"],
                                            r["char_start"], r["char_end"],
                                            len(r["text"])))
    print("pdftotext: %s" % tool)
    for p in per_pdf:
        print("pdf %02d: %6d chars -> %3d chunks  %s" %
              (p["pdf_index"], p["text_chars"], p["chunk_count"], p["path"]))
    print("TOTAL chunks: %d" % len(rows))


if __name__ == "__main__":
    main()
