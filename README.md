# Deterministic Embedding Retrieval Proof — v0.1 PRELIMINARY (corpus v2)

**Status: v0.1 PRELIMINARY / SMOKE TEST — corpus v2 (693 chunks), 2 passes + permutation
test, one VM. This is NOT the final proof.**
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

Ten PDFs, chosen by fixed seed (seed 42) from 29 candidates, broken into 699 chunks,
deduplicated to 693 (corpus v2 — see `DEDUP.md`), and embedded twice with speedrunpapi's
`exact_embed` (deterministic, integer-exact). Each chunk retrieves against all 693 chunks
by cosine similarity, under an order-independent top-1 SET rule. The question under test
is whether the embedding is deterministic (same text → bit-identical vector, every time)
and whether retrieval is stable under corpus permutation (each chunk retrieves itself,
regardless of chunk order).

Methodology: speedrunpapi-style determinism — exact SHA-256 hashes of embedding outputs,
same-bits invariance across runs, identical retrieval rankings. See `REPORT.md` for the
falsifiable claims, the measured verdicts, and the limitations. The original v1 run
(699 chunks, argmax rule, 0.9914) is preserved in the report and in the original
artifacts — including the fact that it falsified the first formulation of the retrieval
claim.

## Result summary [MEASURED] — corpus v2

| Claim | Verdict |
|---|---|
| (A) Two independent passes produce bit-identical embeddings | **PASS** — both passes sha256 `e3e0215c…fb13`, `array_equal` True |
| (B) Retrieval result is invariant under corpus permutation | **PASS** — top-1 set 1.0000 (693/693, all sets singleton); permutation test (seed 20261118) PASS: top-1 sets identical after mapping back, cosine matrix exactly equal, per-chunk vectors 693/693 identical, reordered-back hash == pass-1 hash |
| (C) Mean diagonal cosine > mean off-diagonal cosine | **PASS** — 1.0 vs 0.1700 (diagonal is 1.0 by construction; see REPORT.md for why this claim is weak as formulated). Median set margin 0.3376; min 1.509e-04 (near-duplicate 01-029/07-025) |

## Contents

### Corpus v2 (headline)

| File | What it is | sha256 |
|---|---|---|
| `REPORT.md` | Full report: claims, verdicts, methods, limitations | `99fee2dc…2ddf2535` |
| `DEDUP.md` | The 6 removed chunks, the rule, the reason | `7ee40379…bee993bd` |
| `chunk_table_v2.json` | All 693 chunks with text, offsets, pdf index | `20ec2015…4967bfd6` |
| `chunk_table_v2.csv` | Flat view of the chunk table v2 | `db445956…bfc81d89` |
| `embeddings_pass1_v2.npz` | int16 [693, 384] vectors, pass 1 (array hash `e3e0215c…fb13`) | `8280f9b1…6470d` |
| `embeddings_pass2_v2.npz` | int16 [693, 384] vectors, pass 2 (array hash `e3e0215c…fb13`) | `8280f9b1…6470d` |
| `embeddings_perm_v2.npz` | int16 [693, 384] vectors, permuted-order pass (seed 20261118) | `ff6be727…04e2b9` |
| `embedding_hashes_v2.json` | Per-pass array sha256 + timings | `32a5b837…f01b0d77` |
| `similarity_matrix_v2.npz` | Cosine (float64), int64 dots, norms2, chunk ids | `2e80dbcb…009a4dc` |
| `similarity_matrix_v2.csv` | Cosine matrix, flat | `54f7489d…d34b5e6c` |
| `retrieval_results_v2.json` | Top-1 sets, set margins, argmax diagnostic, NL queries | `61ef65ad…1e8` |
| `permutation_test.json` | Permutation-test result (seed 20261118, both sub-tests) | `3b6d57e1…3231d7e1` |
| `chunk_heatmap_v2.png` | 693×693 chunk similarity heatmap | `d7c4d817…ad6c5` |
| `pdf_heatmap_v2.png` | 10×10 PDF-block mean similarity heatmap | `240b2e5a…9360f` |
| `heatmap_v2.html` | Interactive heatmap (same matrix) | `1d4309dc…af2a4` |

### Corpus v1 (original, preserved for the audit trail)

| File | What it is | sha256 |
|---|---|---|
| `REPORT_v1_original.md` | The original v1 report text, unedited | — |
| `README_v1_original.md` | The original v1 README, unedited | — |
| `corpus-selection.json` | 29 candidates, seed 42, the 10 selected, exclusions | `dcacdbae…a908` |
| `chunk_table.json` | All 699 chunks with text, offsets, pdf index | `f8529d13…197a8` |
| `chunk_table.csv` | Flat view of the chunk table | `12021b8e…7e5c9` |
| `embeddings_pass1.npz` | int16 [699, 384] vectors, pass 1 (array hash `e3fe5b6d…199d`) | `01e5f9ea…405d85` |
| `embeddings_pass2.npz` | int16 [699, 384] vectors, pass 2 (array hash `e3fe5b6d…199d`) | `fb53ddcc…10c316` |
| `embedding_hashes.json` | Per-pass array sha256 + timings (v1) | `f454fcfd…01c2` |
| `similarity_matrix.npz` | Cosine (float64), int64 dots, norms2, chunk ids (v1) | `2a728776…2f95b2` |
| `similarity_matrix.csv` | Cosine matrix, flat (v1) | `54aa00cb…2577` |
| `retrieval_results.json` | Top-1/top-5, margins, failures, NL queries (v1) | `693ac5ee…dcf1` |
| `chunk_heatmap.png` | 699×699 chunk similarity heatmap (v1) | `5310773b…49b2` |
| `pdf_heatmap.png` | 10×10 PDF-block mean similarity heatmap (v1) | `cd853a28…9fde` |
| `heatmap.html` | Interactive heatmap (v1 matrix) | `0fa35d8e…a48d6` |

### Scripts

| File | What it is | sha256 |
|---|---|---|
| `scripts/chunk_pdfs.py` | Corpus chunking (chunking rule v1) | `3f541a30…d6a6ddc4` |
| `scripts/embed_chunks.py` | Two-pass embedding + permuted pass (`--permute-seed`) | `5aef8de2…ac38c88b` |
| `scripts/retrieve.py` | Similarity matrix + top-1 SET retrieval + NL queries | `90b055e1…aaaf6398` |
| `scripts/make_heatmap.py` | Heatmap rendering | `8c96f422…9f71ca05` |
| `scripts/permutation_test.py` | Permutation test (both sub-tests) | `0cdaea25…aa829619` |

## Corpus (seed 42, 10 of 29 candidates)

| # | PDF (path under `~/workspace`) | Chunks (v1 → v2) |
|---|---|---|
| 00 | `technosignature/fixes/drive-copy-B.pdf` | 54 → 54 |
| 01 | `doe-rfi-cryogenic-quantum-testbeds/response-draft-v2-merged.pdf` | 79 → 79 |
| 02 | `browser_downloads/sess-273542462/browser-download-20261117T123612.269045621Z-0-892431-27-RFI-0001_-_RFI_for_Cryogenic_QHPC_Testbed.pdf` | 46 → 46 |
| 03 | `user/files/Monetizing_20Sovereign_20Invariance_20Suite.pdf.pdf` | 11 → 11 |
| 04 | `file-versions/igor-holt-resume-matraix/v1/tree/igor-holt-resume-matraix.pdf` | 15 → 15 |
| 05 | `file-versions/gem-audit-offer/v1/tree/gem-audit-offer.pdf` | 20 → 20 |
| 06 | `user/files/genesis_revenue_plan.pdf` | 94 → 94 |
| 07 | `doe-rfi-cryogenic-quantum-testbeds/response-draft.pdf` | 72 → **66** (6 byte-identical twins of PDF-01 chunks removed) |
| 08 | `your_files/forbes-2026-benchmark.pdf` | 42 → 42 |
| 09 | `quantum-comp/data/verifiable_circuit_compilation.pdf` | 266 → 266 |

Total: 699 chunks (v1) → **693 chunks (v2)**. Full per-PDF sha256 and sizes are in
`corpus-selection.json` and `chunk_table.json` / `chunk_table_v2.json`. Exclusions from
the 29 candidates: 2 browser-page printouts (not papers/docs), 5 duplicate files removed
by sha256, 1 directory entry dropped (sha256 failed).

## Reproduce

Run from this directory. The scripts take arguments (see `--help`); defaults reproduce
the v1 run, `--prefix v2` reproduces the v2 run. The PDFs are read from `~/workspace`
via the paths in `corpus-selection.json`.

```bash
# 1. Chunk the 10 PDFs (system python3; needs pdftotext 24.02.0)
python3 scripts/chunk_pdfs.py

# 2. Embed twice, corpus v2 (venv python: numpy 2.5.5, jax 0.12.1; model params in work/model-cache)
work/venv-jax/bin/python scripts/embed_chunks.py --table chunk_table_v2.json --prefix v2 --passes 2

# 3. Similarity matrix + top-1 SET retrieval + NL queries (system python3, numpy 1.26.4)
python3 scripts/retrieve.py --table chunk_table_v2.json --embeddings embeddings_pass1_v2.npz --prefix v2

# 4. Permuted-order embedding pass (venv python; ~9 min)
work/venv-jax/bin/python scripts/embed_chunks.py --table chunk_table_v2.json --prefix v2 --permute-seed 20261118

# 5. Permutation test (system python3)
python3 scripts/permutation_test.py --table chunk_table_v2.json --embeddings embeddings_pass1_v2.npz --perm-embeddings embeddings_perm_v2.npz --matrix similarity_matrix_v2.npz --seed 20261118

# 6. Heatmaps (SYSTEM python3, numpy 1.26.4 + matplotlib 3.6.3 — the venv python
#    picks up the system matplotlib and fails on the numpy mismatch)
python3 scripts/make_heatmap.py --table chunk_table_v2.json --results retrieval_results_v2.json --hashes embedding_hashes_v2.json --matrix similarity_matrix_v2.npz --prefix v2 --run-label "CORPUS v2 — 2 passes"
```

Re-running the v2 commands on the same inputs must reproduce the chunk table v2 and the
embedding array hash
(`e3e0215c3f192fc8ae54c1fdb7cdb17d39966276cce9b528eef5880bdf0efb13`). Re-running the v1
commands (defaults, no `--prefix`) must reproduce the v1 array hash
(`e3fe5b6d1ca5cd98c467b0c48b09e7abf9d4d33463d752dc0438f1722e48199d`). The speedrunpapi
clone lives in `work/speedrunpapi` (pinned commit `8479566cfd3b008f927184d32ac9b6a98deec3a5`);
the model cache in `work/model-cache`. Neither is part of the published dataset (the
scripts + the pinned commit + the model revision are sufficient to rebuild them).

## Environment [MEASURED]

- Python 3.12.3. System numpy 1.26.4, matplotlib 3.6.3; venv numpy 2.5.5, jax 0.12.1,
  jaxlib 0.12.1, matplotlib 3.6.3, pillow 10.2.0, scipy 1.18.1. pdftotext 24.02.0 (Poppler).
- Model: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Engine: `exact_embed.Exact` (NumPy int64),
  batch 32, int16 output [693, 384] at scale 2^-15.
- Chunking rule v1: `pdftotext -enc UTF-8`, whitespace collapsed to single spaces,
  target 500 chars, 50-char overlap, sentence-boundary snap (last `.` `!` `?` + space in
  [start+250, start+500)), min chunk 100 chars (short final chunk merged into previous).
- Retrieval rule (v2): top-1 SET — all chunks achieving the max cosine (exact float64
  equality); claim = source chunk in its top-1 set. Permutation test seed 20261118.
- Pass times on this VM (CPU, v2): 398.2 s (pass 1), 332.6 s (pass 2), 540.0 s
  (permuted pass). V1 pass times: 399.4 s / 375.7 s.

## Attribution

Embedding engine: [speedrunpapi](https://github.com/lmtlssss/speedrunpapi) (upstream,
lmtlssss); this work uses the fork `igor-holt/speedrunpapi` at pinned commit `8479566…`.
Determinism methodology follows speedrunpapi: exact hashes, same-bits invariance,
identical retrieval rankings.
