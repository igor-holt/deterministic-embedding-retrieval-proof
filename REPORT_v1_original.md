# Deterministic Embedding Retrieval Proof — REPORT (v0.1, smoke test)

**Version:** v0.1 PRELIMINARY / SMOKE TEST — 2 passes, one VM, one model.
**Date:** 2026-11-18 (EST). Smoke-test run completed overnight 2026-11-18 → 2026-11-19 UTC.
**Status:** This is the pipeline validation and the smoke-test result. It is NOT the final
proof. The final proof is the 10,000-repetition run (coordinated separately); its verdict
will replace this one in v1.0. Nothing here is presented as the 10,000-run verdict.

All figures below are [MEASURED] on this VM unless marked otherwise. Where a figure was
recomputed independently of the build scripts (array hashes, similarity margins, twin
checks, NL-query counts), it is marked [MEASURED — re-verified].

## 1. The claims (stated before the results)

Three falsifiable claims, each with its falsification condition:

- **(a) Bit-exact determinism.** Two independent passes over the same 699 chunks produce
  bit-identical embedding vectors. *Falsified by:* any difference between the two passes
  (array inequality, or differing SHA-256 of the embedding bytes).
- **(b) Self-retrieval top-1 = 100%.** For every chunk, querying with that chunk's own text
  retrieves that same chunk at rank 1. *Falsified by:* any query whose top-1 is a different
  chunk.
- **(c) Diagonal dominance.** Mean cosine similarity of each chunk with itself exceeds the
  mean cosine similarity between different chunks. *Falsified by:* mean diagonal ≤ mean
  off-diagonal.

## 2. Verdicts

### (a) Bit-exact determinism — PASS [MEASURED — re-verified]

| | |
|---|---|
| Pass 1 array sha256 | `e3fe5b6d1ca5cd98c467b0c48b09e7abf9d4d33463d752dc0438f1722e48199d` |
| Pass 2 array sha256 | `e3fe5b6d1ca5cd98c467b0c48b09e7abf9d4d33463d752dc0438f1722e48199d` |
| `array_equal` | True |
| Shape / dtype | [699, 384], int16 |

Both passes re-loaded the model parameters from disk and built a fresh engine, then
embedded all 699 chunks in chunk order (batch 32). Pass times: 399.4 s / 375.7 s (CPU).
The two NPZ files have different container hashes (NPZ metadata), but the `vectors`
arrays inside are bit-identical — the array hash above is the hash of the int16 bytes,
computed independently of the build script.

Engine self-check during the build: the exact_embed engine's STS-B reference hash
`031d6912ba5a0824fd2b2562abb592fe5caf8053e0d5a2060cf051a9ab202934` matched its expected
value (the engine refuses to proceed on mismatch). [MEASURED — build-run self-check,
not re-run here.]

### (b) Self-retrieval top-1 = 100% — FAIL as stated [MEASURED — re-verified]

| | |
|---|---|
| Top-1 accuracy | 0.9914 (693/699) |
| Top-5 containment | 1.0 (699/699) |
| Failing chunks | 6: `07-000`, `07-012`, `07-030`, `07-051`, `07-058`, `07-068` |

**The claim as written is falsified.** The mechanism is verified, and it is a corpus
artifact, not an embedding failure:

- Every failing chunk has exactly one byte-identical twin chunk in PDF 01
  (`response-draft-v2-merged.pdf`), while the failing chunk itself lives in PDF 07
  (`response-draft.pdf`) — two versions of the same DOE response whose texts overlap.
  Verified pairs: `07-000`↔`01-000`, `07-012`↔`01-016`, `07-030`↔`01-034`,
  `07-051`↔`01-057`, `07-058`↔`01-063`, `07-068`↔`01-076`. These are the ONLY
  identical-text duplicate pairs in the whole corpus (6 extra chunks beyond first copies).
- Both members of each pair score cosine **exactly 1.0** against the query — a tie. The
  ranking step placed the PDF-01 twin first in all six cases, so top-1 is the twin and
  the self chunk sits at rank 2, also at 1.0.
- The twin is a correct retrieval in every semantic sense (identical text); it is a
  failure only of the claim as literally written ("retrieves *that same chunk*").

**Honest reading:** the embedding did not fail — the corpus contains duplicates and the
claim did not account for them. The fix for v1.0 is to deduplicate the corpus, or to
define top-1 success as "self chunk or its byte-identical twin". Either change must be
declared before the 10,000-run, not after it.

### (c) Diagonal dominance — PASS, but weak as formulated [MEASURED — re-verified]

| | |
|---|---|
| Mean diagonal cosine | 1.0 |
| Mean off-diagonal cosine | 0.1714 (0.17140060886821776) |
| Median top-1 margin (top1 − top2) | 0.336 (0.3362566406393742) |
| Minimum top-1 margin | 0.0 (the six tied duplicates) |

This passes, but the claim is weak as formulated: the diagonal is 1.0 *by construction*
(a vector's cosine with itself is always 1), so "diagonal > off-diagonal" carries almost
no information. The numbers that actually carry signal are the separation: off-diagonal
mean 0.1714, and the median top-1 margin 0.336 — a ranking flips under float noise only
if two similarities are within ~1e-15, so a 0.336 median margin means the rankings are
not float artefacts. The minimum margin 0.0 is exactly the six tied duplicates from
claim (b). For v1.0 this claim should be restated in terms of the margin distribution.

## 3. Methods

### Corpus selection
29 PDF candidates enumerated under `~/workspace` (papers/docs only; browser-page
printouts and duplicates excluded — see `corpus-selection.json`). 10 selected by fixed
seed 42 via Python `random`. Selection is reproducible: same seed + same candidate list
→ same 10 PDFs. Per-PDF sha256 and sizes are in `corpus-selection.json`.

### Chunking (rule v1)
`pdftotext -enc UTF-8` (Poppler 24.02.0), whitespace collapsed to single spaces, target
500 chars, 50-char overlap, sentence-boundary snap (last `.` `!` `?` + space in
[start+250, start+500)), min chunk 100 chars (short final chunk merged into previous).
699 chunks total. Per-PDF counts: 00:54, 01:79, 02:46, 03:11, 04:15, 05:20, 06:94,
07:72, 08:42, 09:266. Chunk ids are `<pdf_index:02d>-<chunk_index:03d>` in corpus order.

### Embedding
speedrunpapi `exact_embed` (NumPy int64 exact engine), pinned commit
`8479566cfd3b008f927184d32ac9b6a98deec3a5`, model
`sentence-transformers/all-MiniLM-L6-v2` revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, batch 32, output int16 [699, 384] at scale
2^-15. Two independent passes, each re-loading parameters and building a fresh engine.
Similarity: int64 dot matrix `S = V @ V.T` (exact, order-independent); cosine from
`S[i,j] / sqrt(S[i,i]·S[j,j])` in float64 — same integers in, same float out, on any
machine. No BLAS float summation in the similarity itself.

### Retrieval
Self-retrieval: query = chunk i's own embedding row; rank all 699 chunks by cosine.
Plus 20 hand-written natural-language queries (2 per PDF, verbatim in
`retrieval_results.json`), recorded with top-5 chunk ids and source PDFs.

## 4. Additional results

**Natural-language queries (descriptive only, no accuracy claim attached).**
10 of 20 queries land top-1 in their intended PDF [MEASURED — re-verified]. The
"intended" PDF is the author's judgment, and the queries were written after seeing the
corpus — this number is a sanity check on semantic signal, not a benchmark. Example of
a miss: "How can waste heat reveal an engineered technosignature?" (intended PDF 00)
retrieves PDF 05 top-1 at 0.495, with PDF 00 chunks at ranks 2–3 (0.491, 0.464).

**Heatmaps.** `chunk_heatmap.png` is the full 699×699 cosine matrix (viridis, −0.1 to
1.0) with PDF-block boundaries; `pdf_heatmap.png` is the 10×10 block-mean matrix;
`heatmap.html` is the same matrix interactively. The diagonal is bright throughout;
the visible off-diagonal structure is the DOE-response block (PDFs 01/02/07 share
subject matter) and the quantum-comp paper's internal structure.

## 5. Limitations (read before citing this)

1. **2 passes, not 10,000.** This is a smoke test on one VM. Bit-exactness across two
   passes is evidence, not a 10,000-repetition proof. The 10,000-run verdict supersedes
   this one.
2. **One model, one machine.** Only `exact_embed` / all-MiniLM-L6-v2, only this VM's
   CPU. Determinism on other hardware, other batch sizes, or other models is not
   established here.
3. **Claim (b) is falsified as written** (see §2). The duplicate-corpus mechanism is
   verified, but it is an explanation, not a pass. v1.0 must fix the claim or the corpus
   before the run.
4. **Claim (c) is weak as formulated** (diagonal is 1.0 by construction). The margin
   distribution is the real signal.
5. **NL queries are illustrative.** 10/20 is descriptive; the queries were hand-written
   with knowledge of the corpus and the "intended" PDF is a judgment call.
6. **Corpus is the author's own files.** Ten PDFs from one workspace, not a held-out or
   adversarial corpus. Two are versions of the same document (the source of the claim-(b)
   ties).
7. **Engine self-check not re-run here.** The STS-B hash match is the build run's
   self-check, reported by that run; this report's independent re-verification covers the
   embedding arrays, the similarity matrix, the margins, the twins, and the NL counts.

## 6. What v1.0 (10,000-run) must change

- Deduplicate the corpus (drop byte-identical chunks) or restate claim (b) to accept
  byte-identical twins — declared before the run.
- Restate claim (c) in terms of the top-1 margin distribution.
- 10,000 independent repetitions, sharded across zones/hosts, recording the embedding
  array sha256 (and retrieval ranking) per run; verdict = count of identical vs
  differing runs.

## 7. Reproduce

See `README.md` for exact commands, versions, and file hashes. Re-running on the same
inputs must reproduce the chunk table and the embedding array hash
`e3fe5b6d1ca5cd98c467b0c48b09e7abf9d4d33463d752dc0438f1722e48199d`.
