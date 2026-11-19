# Deterministic Embedding Retrieval Proof — v0.1 PRELIMINARY (smoke test)

**Status: v0.1 PRELIMINARY / SMOKE TEST — 2 passes, one VM. This is NOT the final proof.**
The final proof is the 10,000-repetition run (coordinated separately); this package is the
pipeline, the corpus, and the smoke-test result it was validated on. Nothing in here is
presented as the 10,000-run verdict.

## Where the files live

This package is published in two places with identical text artifacts:

- **Hugging Face dataset** — `igorholt/deterministic-embedding-retrieval-proof`
  (https://huggingface.co/datasets/igorholt/deterministic-embedding-retrieval-proof):
  the complete package, including the binary artifacts (embedding NPZs, similarity NPZ,
  heatmap PNGs).
- **GitHub repo** — `igor-holt/deterministic-embedding-retrieval-proof`
  (https://github.com/igor-holt/deterministic-embedding-retrieval-proof): the text
  artifacts and scripts only. The GitHub connector cannot push binary files, so the
  binary artifacts (embeddings, similarity matrix NPZ, heatmap PNGs) live on the
  Hugging Face dataset linked above.

## License

CC BY 4.0. The embedding engine is speedrunpapi (upstream lmtlssss); see Attribution below.

## What this is

Ten PDFs, chosen by fixed seed (seed 42) from 29 candidates, broken into 699 chunks and
embedded twice with speedrunpapi's `exact_embed` (deterministic, integer-exact). Each chunk
retrieves against all 699 chunks by cosine similarity. The question under test is whether
the embedding is deterministic (same text → bit-identical vector, every time) and whether
retrieval is stable (each chunk retrieves itself).

Methodology: speedrunpapi-style determinism — exact SHA-256 hashes of embedding outputs,
same-bits invariance across runs, identical retrieval rankings. See `REPORT.md` for the
falsifiable claims, the measured verdicts, and the limitations.

## Result summary [MEASURED]

| Claim | Verdict |
|---|---|
| (a) Two independent passes produce bit-identical embeddings | **PASS** — both passes sha256 `e3fe5b6d…199d`, `array_equal` True |
| (b) Self-retrieval top-1 accuracy = 100% | **FAIL as stated** — 0.9914 (693/699). All 6 failures are exact ties (cosine 1.0) with a byte-identical twin chunk from a duplicate PDF; top-5 containment 100% |
| (c) Mean diagonal cosine > mean off-diagonal cosine | **PASS** — 1.0 vs 0.1714 (diagonal is 1.0 by construction; see REPORT.md for why this claim is weak as formulated) |

## Contents

| File | What it is | sha256 |
|---|---|---|
| `REPORT.md` | Full report: claims, verdicts, methods, limitations | — |
| `corpus-selection.json` | 29 candidates, seed 42, the 10 selected, exclusions | `dcacdbae…a908` |
| `chunk_table.json` | All 699 chunks with text, offsets, pdf index | `f8529d13…197a8` |
| `chunk_table.csv` | Flat view of the chunk table | `12021b8e…7e5c9` |
| `embeddings_pass1.npz` | int16 [699, 384] vectors, pass 1 (file hash; array hash `e3fe5b6d…199d`) | `01e5f9ea…405d85` |
| `embeddings_pass2.npz` | int16 [699, 384] vectors, pass 2 (file hash; array hash `e3fe5b6d…199d`) | `fb53ddcc…10c316` |
| `embedding_hashes.json` | Per-pass array sha256 + timings | `f454fcfd…01c2` |
| `similarity_matrix.npz` | Cosine (float64), int64 dots, norms2, chunk ids | `2a728776…2f95b2` |
| `similarity_matrix.csv` | Cosine matrix, flat | `54aa00cb…2577` |
| `retrieval_results.json` | Top-1/top-5, margins, failures, NL queries | `693ac5ee…dcf1` |
| `chunk_heatmap.png` | 699×699 chunk similarity heatmap | `5310773b…49b2` |
| `pdf_heatmap.png` | 10×10 PDF-block mean similarity heatmap | `cd853a28…9fde` |
| `heatmap.html` | Interactive heatmap (same matrix) | `0fa35d8e…a48d6` |
| `scripts/chunk_pdfs.py` | Corpus chunking (chunking rule v1) | `3f541a30…ddc4` |
| `scripts/embed_chunks.py` | Two-pass embedding | `3f1e9bc9…d34ba31` |
| `scripts/retrieve.py` | Similarity matrix + self-retrieval + NL queries | `e9d5c601…198` |
| `scripts/make_heatmap.py` | Heatmap rendering | `1178bc78…b75` |

## Corpus (seed 42, 10 of 29 candidates)

| # | PDF (path under `~/workspace`) | Chunks |
|---|---|---|
| 00 | `technosignature/fixes/drive-copy-B.pdf` | 54 |
| 01 | `doe-rfi-cryogenic-quantum-testbeds/response-draft-v2-merged.pdf` | 79 |
| 02 | `browser_downloads/sess-273542462/browser-download-20261117T123612.269045621Z-0-892431-27-RFI-0001_-_RFI_for_Cryogenic_QHPC_Testbed.pdf` | 46 |
| 03 | `user/files/Monetizing_20Sovereign_20Invariance_20Suite.pdf.pdf` | 11 |
| 04 | `file-versions/igor-holt-resume-matraix/v1/tree/igor-holt-resume-matraix.pdf` | 15 |
| 05 | `file-versions/gem-audit-offer/v1/tree/gem-audit-offer.pdf` | 20 |
| 06 | `user/files/genesis_revenue_plan.pdf` | 94 |
| 07 | `doe-rfi-cryogenic-quantum-testbeds/response-draft.pdf` | 72 |
| 08 | `your_files/forbes-2026-benchmark.pdf` | 42 |
| 09 | `quantum-comp/data/verifiable_circuit_compilation.pdf` | 266 |

Total: 699 chunks. Full per-PDF sha256 and sizes are in `corpus-selection.json` and
`chunk_table.json`. Exclusions from the 29 candidates: 2 browser-page printouts (not
papers/docs), 5 duplicate files removed by sha256, 1 directory entry dropped (sha256 failed).

## Reproduce

Run from this directory. The scripts take no arguments; paths are relative to the
package and the PDFs are read from `~/workspace` via the paths in `corpus-selection.json`.

```bash
# 1. Chunk the 10 PDFs (system python3; needs pdftotext 24.02.0)
python3 scripts/chunk_pdfs.py

# 2. Embed twice (venv python: numpy 2.5.5, jax 0.12.1; model params in work/model-cache)
work/venv-jax/bin/python scripts/embed_chunks.py

# 3. Similarity matrix + self-retrieval + NL queries (system python3, numpy 1.26.4)
python3 scripts/retrieve.py

# 4. Heatmaps (venv python: matplotlib 3.6.3)
work/venv-jax/bin/python scripts/make_heatmap.py
```

Re-running on the same inputs must reproduce the same chunk table and the same embedding
array hash (`e3fe5b6d1ca5cd98c467b0c48b09e7abf9d4d33463d752dc0438f1722e48199d`). The
speedrunpapi clone lives in `work/speedrunpapi` (pinned commit
`8479566cfd3b008f927184d32ac9b6a98deec3a5`); the model cache in `work/model-cache`.
Neither is part of the published dataset (the scripts + the pinned commit + the model
revision are sufficient to rebuild them).

## Environment [MEASURED]

- Python 3.12.3. System numpy 1.26.4; venv numpy 2.5.5, jax 0.12.1, jaxlib 0.12.1,
  matplotlib 3.6.3, pillow 10.2.0, scipy 1.18.1. pdftotext 24.02.0 (Poppler).
- Model: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Engine: `exact_embed.Exact` (NumPy int64),
  batch 32, int16 output [699, 384] at scale 2^-15.
- Chunking rule v1: `pdftotext -enc UTF-8`, whitespace collapsed to single spaces,
  target 500 chars, 50-char overlap, sentence-boundary snap (last `.` `!` `?` + space in
  [start+250, start+500)), min chunk 100 chars (short final chunk merged into previous).
- Pass times on this VM (CPU): 399.4 s (pass 1), 375.7 s (pass 2).

## Attribution

Embedding engine: [speedrunpapi](https://github.com/lmtlssss/speedrunpapi) (upstream,
lmtlssss); this work uses the fork `igor-holt/speedrunpapi` at pinned commit `8479566…`.
Determinism methodology follows speedrunpapi: exact hashes, same-bits invariance,
identical retrieval rankings.
