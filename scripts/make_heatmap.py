#!/usr/bin/env python3
"""Render the retrieval similarity matrix as heatmaps (PNG + HTML).

Two figures:
  1. chunk_heatmap[_<prefix>].png — full NxN query-chunk cosine matrix.
     Axes are chunk ids; PDF block boundaries are drawn as lines; the
     diagonal (self-similarity) is marked. Title carries the measured
     top-1 accuracy and N.
  2. pdf_heatmap[_<prefix>].png — 10x10 mean cosine between the chunks of
     each pair of PDFs (block means of the chunk matrix). The readable
     summary view.
Also writes heatmap[_<prefix>].html: self-contained (PNGs embedded as
base64) with the caption, the measured numbers, and the falsifiable-claims
verdicts.

Options (defaults reproduce the v1 figures exactly):
  --table PATH      chunk table JSON (default: chunk_table.json)
  --results PATH    retrieval results JSON (default: retrieval_results.json)
  --hashes PATH     embedding hashes JSON (default: embedding_hashes.json)
  --matrix PATH     similarity matrix NPZ (default: similarity_matrix.npz)
  --prefix STR      output prefix, e.g. v2 (default: "")
  --run-label STR   label line 1 of every title (default: SMOKE TEST — 2 passes)
"""
import argparse
import base64
import io
import json
import os
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROOF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
MODEL_LABEL = "exact_embed MiniLM-L6-v2 (int16)"
SHARD_LABEL = "local VM, single shard"


def b64_png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="chunk_table.json")
    ap.add_argument("--results", default="retrieval_results.json")
    ap.add_argument("--hashes", default="embedding_hashes.json")
    ap.add_argument("--matrix", default="similarity_matrix.npz")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--run-label", default="SMOKE TEST — 2 passes")
    args = ap.parse_args()
    suf = ("_" + args.prefix) if args.prefix else ""
    RUN_LABEL = args.run_label

    table = json.load(open(os.path.join(PROOF, args.table)))
    res = json.load(open(os.path.join(PROOF, args.results)))
    hashes = json.load(open(os.path.join(PROOF, args.hashes)))
    d = np.load(os.path.join(PROOF, args.matrix))
    C = d["cosine"]
    ids = [str(x) for x in d["chunk_ids"]]
    pdf_of = np.array([c["pdf_index"] for c in table["chunks"]])
    N = len(ids)
    bounds = []
    for p in range(10):
        idx = np.where(pdf_of == p)[0]
        if len(idx):
            bounds.append((idx[0], idx[-1] + 1))
    labels = ["%02d %s" % (p["pdf_index"], os.path.basename(p["path"])[:28])
              for p in table["per_pdf"]]
    # top-1 numbers: v2 results carry the SET rule; v1 carries the argmax rule
    if "top1_set_accuracy" in res:
        top1_acc = res["top1_set_accuracy"]
        top1_correct = res["top1_set_correct"]
        top1_note = "top-1 SET accuracy (order-independent rule)"
        set_sizes = res.get("top1_set_sizes", {})
        verdict_b = "PASS" if res["top1_set_accuracy"] == 1.0 else "FAIL"
        margin = res.get("min_set_margin", res["min_top1_margin"])
    else:
        top1_acc = res["top1_accuracy"]
        top1_correct = res["top1_correct"]
        top1_note = "top-1 accuracy (argmax-first rule)"
        set_sizes = {}
        verdict_b = "PASS" if res["top1_accuracy"] == 1.0 else "FAIL"
        margin = res["min_top1_margin"]

    # --- figure 1: full chunk matrix ---
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(C, cmap="viridis", vmin=-0.1, vmax=1.0, aspect="auto")
    for a, b in bounds:
        ax.axhline(a - 0.5, color="white", lw=0.8)
        ax.axvline(a - 0.5, color="white", lw=0.8)
    ax.set_title("%s | %s | %s | %s\n"
                 "Self-retrieval cosine similarity — %d chunks from 10 PDFs\n"
                 "%s %.4f (%d/%d)  |  mean diagonal %.4f  |  "
                 "mean off-diagonal %.4f"
                 % (RUN_LABEL, MODEL_LABEL, SHARD_LABEL, STAMP,
                    N, top1_note, top1_acc, top1_correct, N,
                    res["mean_diagonal_cosine"], res["mean_offdiagonal_cosine"]),
                 fontsize=10)
    ax.set_xlabel("retrieved chunk id (in corpus order)")
    ax.set_ylabel("query chunk id (in corpus order)")
    ax.set_xticks([b[0] for b in bounds])
    ax.set_xticklabels(["%02d" % i for i in range(len(bounds))], fontsize=8)
    ax.set_yticks([b[0] for b in bounds])
    ax.set_yticklabels(["%02d" % i for i in range(len(bounds))], fontsize=8)
    fig.colorbar(im, ax=ax, label="cosine similarity")
    fig.tight_layout()
    fig.savefig(os.path.join(PROOF, "chunk_heatmap%s.png" % suf), dpi=150, bbox_inches="tight")
    b64_chunk = b64_png(fig)

    # --- figure 2: PDF block means ---
    M = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            M[i, j] = C[np.ix_(pdf_of == i, pdf_of == j)].mean()
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    im2 = ax2.imshow(M, cmap="viridis", vmin=0.0, vmax=1.0)
    ax2.set_xticks(range(10))
    ax2.set_yticks(range(10))
    ax2.set_xticklabels(["%02d" % i for i in range(10)])
    ax2.set_yticklabels(["%02d" % i for i in range(10)])
    ax2.set_xlabel("retrieved PDF")
    ax2.set_ylabel("query PDF")
    ax2.set_title("%s | %s | %s | %s\n"
                  "Mean chunk cosine by PDF pair (block means of the %dx%d matrix)"
                  % (RUN_LABEL, MODEL_LABEL, SHARD_LABEL, STAMP, N, N), fontsize=10)
    for i in range(10):
        for j in range(10):
            ax2.text(j, i, "%.2f" % M[i, j], ha="center", va="center",
                     fontsize=8, color="white" if M[i, j] < 0.5 else "black")
    fig2.colorbar(im2, ax=ax2, label="mean cosine similarity")
    fig2.tight_layout()
    fig2.savefig(os.path.join(PROOF, "pdf_heatmap%s.png" % suf), dpi=150, bbox_inches="tight")
    b64_pdf = b64_png(fig2)

    def count_of(p):
        return p.get("chunk_count_v2", p.get("chunk_count"))
    legend = "\n".join("<li><b>%02d</b> — %s (%d chunks)</li>" %
                       (p["pdf_index"], p["path"], count_of(p))
                       for p in table["per_pdf"])
    rule_note = ("Top-1 is scored as a SET: the source chunk must be among the "
                 "chunks achieving the maximum cosine in its row. The set is "
                 "order-independent by construction. Set sizes: %s." % set_sizes) if set_sizes else (
        "Top-1 is scored argmax-first (v1 rule): exact ties are broken by corpus "
        "order. 6 of 699 queries retrieve a byte-identical twin chunk from the "
        "other version of the same DOE response (PDFs 01 and 07 share 6 identical "
        "chunks) instead of their own chunk id — top-1 accuracy 0.9914 as stated; "
        "top-1 text-identical accuracy 1.0000. See REPORT.md for the full failure "
        "analysis.")
    html = """<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Deterministic embedding — semantic retrieval heatmap</title>
<style>body{font-family:system-ui,sans-serif;max-width:1000px;margin:2em auto;padding:0 1em}
img{max-width:100%%} table{border-collapse:collapse} td,th{border:1px solid #ccc;padding:4px 8px;text-align:left}</style>
</head><body>
<h1>Deterministic embedding — semantic retrieval heatmap</h1>
<p><b>%s | %s | %s | %s</b></p>
<p>%d chunks from 10 randomly sampled PDFs (seed 42), embedded with
speedrunpapi exact_embed (all-MiniLM-L6-v2, int16, bit-exact). Each row is a
query chunk; each column is a retrieved chunk; colour is cosine similarity.
The bright diagonal is self-retrieval. %s</p>
<h2>Measured results</h2>
<table>
<tr><th>Claim</th><th>Measured</th><th>Verdict</th></tr>
<tr><td>(A) Pass 1 == Pass 2 bit-exact (embedding determinism)</td><td>pass1 %s<br>pass2 %s</td><td>%s</td></tr>
<tr><td>(B) Retrieval determinism — source chunk in top-1 set</td><td>%.4f (%d/%d)</td><td>%s</td></tr>
<tr><td>(C) Mean diagonal vs off-diagonal cosine</td><td>%.4f vs %.4f</td><td>%s</td></tr>
</table>
<p>Min set margin (max &minus; best non-tied cosine): %.3e — the top-1 set is
not a float artefact (a flip needs two similarities within ~1e-15).</p>
<h2>Full chunk matrix (%d &times; %d)</h2>
<img src="data:image/png;base64,%s" alt="chunk similarity heatmap">
<h2>PDF-level mean similarity (10 &times; 10)</h2>
<img src="data:image/png;base64,%s" alt="pdf similarity heatmap">
<h2>PDF legend</h2><ul>%s</ul>
<p>Raw data: similarity_matrix%s.csv / similarity_matrix%s.npz.
Reproduce: see README.md in the same package.</p>
</body></html>""" % (
        RUN_LABEL, MODEL_LABEL, SHARD_LABEL, STAMP, N, rule_note,
        hashes["pass1_sha256"][:16] + "…", hashes["pass2_sha256"][:16] + "…",
        "PASS" if hashes["bit_identical"] else "FAIL",
        top1_acc, top1_correct, N, verdict_b,
        res["mean_diagonal_cosine"], res["mean_offdiagonal_cosine"],
        "PASS" if res["mean_diagonal_cosine"] > res["mean_offdiagonal_cosine"] else "FAIL",
        margin, N, N, b64_chunk, b64_pdf, legend, suf, suf)
    open(os.path.join(PROOF, "heatmap%s.html" % suf), "w").write(html)
    print("wrote chunk_heatmap%s.png, pdf_heatmap%s.png, heatmap%s.html" % (suf, suf, suf))


if __name__ == "__main__":
    main()
