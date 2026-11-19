# Deterministic Embedding Retrieval Proof — REPORT (v0.1, corpus v2)

**Version:** v0.1 PRELIMINARY / SMOKE TEST — corpus v2 (693 chunks), 2 passes + permutation
test, one VM, one model.
**Date:** 2026-11-18 (EST). v1 run overnight 2026-11-18 → 2026-11-19 UTC; v2 run
2026-11-19 ~04:10–04:40 UTC.
**Status:** This is the pipeline validation and the smoke-test result. It is NOT the final
proof. The final proof is the 10,000-repetition run (coordinated separately); its verdict
will replace this one in v1.0. Nothing here is presented as the 10,000-run verdict.

**Headline (corpus v2):** embedding determinism PASS (bit-identical across 2 passes);
retrieval determinism PASS (top-1 set 693/693 + permutation test PASS). The original v1
run (699 chunks) is preserved in full below and in the original artifacts — including
the fact that it falsified the first formulation of the retrieval claim.

All figures below are [MEASURED] on this VM unless marked otherwise. Where a figure was
recomputed independently of the build scripts (array hashes, similarity margins, twin
checks, set-rule counts, NL-query counts), it is marked [MEASURED — re-verified].

## 1. The claims (split in two — see §4 for why)

Determinism splits into two claims, each with its own falsification condition:

- **(A) Embedding determinism.** Two independent passes over the same chunks produce
  bit-identical embedding vectors. *Falsified by:* any difference between the two passes
  (array inequality, or differing SHA-256 of the embedding bytes).
- **(B) Retrieval determinism.** The retrieval result does not depend on corpus order.
  Operationalized as: (i) the top-1 SET rule — `top1_set[i] = { j : cosine[i,j] ==
  max_k cosine[i,k] }` (exact float64 equality) — with the claim that the source chunk
  `i` is always in `top1_set[i]`; and (ii) a permutation test — shuffling the chunk
  order with a fixed recorded seed and re-running retrieval leaves the result unchanged.
  *Falsified by:* any source chunk outside its top-1 set, or any change in the result
  under permutation.
- **(C) Diagonal dominance.** Mean cosine similarity of each chunk with itself exceeds
  the mean cosine similarity between different chunks. *Falsified by:* mean diagonal ≤
  mean off-diagonal. (Weak as formulated — the diagonal is 1.0 by construction; the
  margin distribution carries the real signal. See §2.)

## 2. Verdicts — corpus v2 (headline)

### (A) Embedding determinism — PASS [MEASURED — re-verified]

| | |
|---|---|
| Pass 1 array sha256 | `e3e0215c3f192fc8ae54c1fdb7cdb17d39966276cce9b528eef5880bdf0efb13` |
| Pass 2 array sha256 | `e3e0215c3f192fc8ae54c1fdb7cdb17d39966276cce9b528eef5880bdf0efb13` |
| `array_equal` | True |
| Shape / dtype | [693, 384], int16 |

Both passes re-loaded the model parameters from disk and built a fresh engine, then
embedded all 693 chunks in chunk order (batch 32). Pass times: 398.2 s / 332.6 s (CPU).
The two NPZ files are byte-identical as containers, too (same file sha256
`8280f9b1…6470d`) — the array hash above is the hash of the int16 bytes, computed
independently of the build script.

Engine self-check during the build: the exact_embed engine's STS-B reference hash
`031d6912ba5a0824fd2b2562abb592fe5caf8053e0d5a2060cf051a9ab202934` matched its expected
value (the engine refuses to proceed on mismatch). [MEASURED — build-run self-check,
not re-run here.]

### (B) Retrieval determinism — PASS [MEASURED — re-verified]

| | |
|---|---|
| Top-1 SET accuracy | 1.0000 (693/693) |
| Top-1 set sizes | {1: 693} — every set is a singleton |
| Top-1 set failures | none |
| Diagnostic argmax-first top-1 | 1.0000 (693/693), no failures |
| Top-5 containment | 1.0000 (693/693) |
| Permutation test (seed 20261118) | **PASS** (both sub-tests, see §5) |

The source chunk is in its top-1 set for all 693 queries, and every set has exactly one
member — so on this corpus the set rule and the argmax rule agree completely. The
permutation test (§5) confirms the result does not depend on corpus order.

### (C) Diagonal dominance — PASS, but weak as formulated [MEASURED — re-verified]

| | |
|---|---|
| Mean diagonal cosine | 1.0 |
| Mean off-diagonal cosine | 0.1700 (0.16997688301673872) |
| Median set margin | 0.3376 (0.33756615692099423) |
| Minimum set margin | 1.509e-04, at chunk 01-029 (runner-up 07-025 at 0.999849) |

The diagonal is 1.0 by construction, so the claim itself is weak; the signal is the
separation. The median set margin 0.3376 is far above float noise (~1e-15), so the
rankings are not float artefacts. The minimum margin 1.509e-04 is the closest call in
the corpus: chunks 01-029 and 07-025 are near-duplicates (overlapping DOE-response
passages that differ slightly, so they survived the byte-identical dedup). Four rows
have a margin below 1e-3 (01-029, 01-066, 07-025, 07-061) — all cross-version
near-duplicates between PDFs 01 and 07. Even the smallest margin is ~11 orders of
magnitude above float noise.

## 3. The v1 run (original, preserved)

The first run used corpus v1: 699 chunks, argmax-first retrieval rule. It **falsified**
the retrieval claim as first formulated, and that falsification is the reason this
report exists in its current form. Nothing about the v1 run has been altered; its
artifacts (`chunk_table.json`, `embeddings_pass1/2.npz`, `similarity_matrix.npz`,
`chunk_heatmap.png`, `pdf_heatmap.png`, `heatmap.html`, `retrieval_results.json`) are
preserved in the package, and the v1 report text is preserved as `REPORT_v1_original.md`.

### v1 verdicts [MEASURED — re-verified]

- **(A) Embedding determinism — PASS.** Both passes sha256
  `e3fe5b6d1ca5cd98c467b0c48b09e7abf9d4d33463d752dc0438f1722e48199d`, `array_equal`
  True. Pass times 399.4 s / 375.7 s (CPU). [693-chunk note: the v1 array hash differs
  from the v2 hash because the corpus differs (699 vs 693 chunks).]
- **(b) Self-retrieval top-1 (argmax-first) = 100% — FAIL as stated.** 0.9914 (693/699).
  All 6 failures were PDF-07 chunks retrieving their byte-identical twin in PDF 01 at
  cosine exactly 1.0 — a tie, broken by corpus order: `07-000`→`01-000`,
  `07-012`→`01-016`, `07-030`→`01-034`, `07-051`→`01-057`, `07-058`→`01-063`,
  `07-068`→`01-076`. These were the ONLY identical-text duplicate pairs in the corpus.
  Top-5 containment 1.0 (699/699). Mean diagonal 1.0, off-diagonal 0.1714.
- **v1 under the SET rule (recomputed from the v1 matrix): 699/699**, set sizes
  {1: 687, 2: 12} — the 12 two-element sets are the 6 failing chunks plus their 6
  twins. [MEASURED — re-verified.] The set rule would have passed on the v1 data; the
  argmax rule failed. That is the whole point: the rule, not the embedding, was
  order-dependent.

## 4. The fix — why two changes were needed

Igor's challenge ("that is not deterministic then.. is it") was correct, and the fix
had to go deeper than the dedup:

- **Embedding determinism was already proven** (bit-identical vectors across two passes).
  That claim stands untouched.
- **Retrieval determinism was NOT proven as implemented.** Exact ties were broken by
  corpus order (argmax returns the first maximum), so permuting the corpus flips the
  top-1 answer between twins. The dedup removes the ties in *this* corpus but does not
  make the *rule* deterministic — a future corpus with one exact tie would fail the
  same way.

So the fix is two parts:

1. **Dedup (corpus v2, 693 chunks).** Remove the second occurrence in corpus order of
   any byte-identical chunk text; keep the first. Removed: the 6 PDF-07 chunks above.
   Full record: `DEDUP.md`. Twin-text equality was verified programmatically before
   removal. Chunk IDs are stable (the remaining PDF-07 chunks keep their original IDs;
   the removed IDs are absent by design).
2. **Order-independent retrieval rule (top-1 SET).** `top1_set[i] = { j : cosine[i,j] ==
   max_k cosine[i,k] }`. The set does not depend on corpus order: permuting the corpus
   permutes the set's labels the same way and never changes membership. The alternative
   considered — tie-breaking by chunk content hash — was rejected: identical-text twins
   have identical content hashes, so it cannot break exactly the tie it was designed
   for. The set rule handles any future tie gracefully. Both the dedup and the rule
   change were declared before the v2 run, not after it.

## 5. Permutation test (seed 20261118) — PASS [MEASURED]

The falsification test for retrieval determinism. Seed 20261118, recorded in
`permutation_test.json` and in `scripts/permutation_test.py`.

- **Test 1 — ranking implementation.** Permuted the v2 pass-1 embedding rows with the
  seed, recomputed the cosine matrix, computed top-1 sets in permuted coordinates,
  mapped them back to chunk ids. Result: top-1 sets identical after mapping back
  (**True**); un-permuted cosine matrix exactly equal to the v2 matrix (**True**,
  elementwise float64 equality).
- **Test 2 — embedding order-independence.** A third embedding pass ran the same 693
  chunks in shuffled order (seed 20261118, different batch composition, 540.0 s).
  Result: per-chunk vectors identical **693/693**; reordered-back array hash equals
  the pass-1 hash (**True**, `e3e0215c…fb13`). The embedding does not depend on chunk
  order or batch composition.
- **Verdict: PASS** (both sub-tests).

## 6. Methods

### Corpus selection
29 PDF candidates enumerated under `~/workspace` (papers/docs only; browser-page
printouts and duplicates excluded — see `corpus-selection.json`). 10 selected by fixed
seed 42 via Python `random`. Selection is reproducible: same seed + same candidate list
→ same 10 PDFs. Per-PDF sha256 and sizes are in `corpus-selection.json`.

### Chunking (rule v1)
`pdftotext -enc UTF-8` (Poppler 24.02.0), whitespace collapsed to single spaces, target
500 chars, 50-char overlap, sentence-boundary snap (last `.` `!` `?` + space in
[start+250, start+500)), min chunk 100 chars (short final chunk merged into previous).
699 chunks in v1; 693 in v2 (PDF 07: 72 → 66, all others unchanged). Per-PDF v2 counts:
00:54, 01:79, 02:46, 03:11, 04:15, 05:20, 06:94, 07:66, 08:42, 09:266. Chunk ids are
`<pdf_index:02d>-<chunk_index:03d>` in corpus order.

### Embedding
speedrunpapi `exact_embed` (NumPy int64 exact engine), pinned commit
`8479566cfd3b008f927184d32ac9b6a98deec3a5`, model
`sentence-transformers/all-MiniLM-L6-v2` revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, batch 32, output int16 [N, 384] at scale
2^-15. Each pass re-loads parameters and builds a fresh engine. Similarity: int64 dot
matrix `S = V @ V.T` (exact, order-independent); cosine from
`S[i,j] / sqrt(S[i,i]·S[j,j])` in float64 — same integers in, same float out, on any
machine. No BLAS float summation in the similarity itself.

### Retrieval (v2 rule)
Self-retrieval: query = chunk i's own embedding row; top-1 SET as defined in §1. The
old argmax-first rule is still computed and recorded as a diagnostic
(`argmax_top1_*` in `retrieval_results_v2.json`) so the two rules can be compared.
Plus 20 hand-written natural-language queries (2 per PDF, verbatim in
`retrieval_results_v2.json`), recorded with top-5 chunk ids and source PDFs.

## 7. Additional results

**Natural-language queries (descriptive only, no accuracy claim attached).**
10 of 20 queries land top-1 in their intended PDF, on both v1 and v2
[MEASURED — re-verified]. The "intended" PDF is the author's judgment, and the queries
were written after seeing the corpus — this number is a sanity check on semantic
signal, not a benchmark. Example of a miss: "How can waste heat reveal an engineered
technosignature?" (intended PDF 00) retrieves PDF 05 top-1 at 0.495, with PDF 00
chunks at ranks 2–3 (0.491, 0.464).

**Heatmaps.** `chunk_heatmap_v2.png` is the full 693×693 cosine matrix (viridis, −0.1
to 1.0) with PDF-block boundaries; `pdf_heatmap_v2.png` is the 10×10 block-mean matrix;
`heatmap_v2.html` is the same matrix interactively. The diagonal is bright throughout;
the visible off-diagonal structure is the DOE-response block (PDFs 01/02/07 share
subject matter) and the quantum-comp paper's internal structure. The v1 heatmaps
(`chunk_heatmap.png`, `pdf_heatmap.png`, `heatmap.html`) are preserved alongside.

## 8. Limitations (read before citing this)

1. **2 passes + 1 permuted pass, not 10,000.** This is a smoke test on one VM.
   Bit-exactness across two passes and permutation-invariance on one shuffle are
   evidence, not a 10,000-repetition proof. The 10,000-run verdict supersedes this one.
2. **One model, one machine, one shuffle seed.** Only `exact_embed` / all-MiniLM-L6-v2,
   only this VM's CPU, one permutation seed (20261118). Determinism on other hardware,
   other batch sizes, other models, or other shuffles is not established here.
3. **Claim (C) is weak as formulated** (diagonal is 1.0 by construction). The margin
   distribution is the real signal.
4. **Closest call is a near-duplicate.** Minimum set margin 1.509e-04 (01-029 vs 07-025)
   is far above float noise but shows the two DOE versions share near-identical passages
   that byte-identical dedup cannot remove. A corpus with a true near-duplicate tie
   (margin below ~1e-12) would need the same treatment.
5. **NL queries are illustrative.** 10/20 is descriptive; the queries were hand-written
   with knowledge of the corpus and the "intended" PDF is a judgment call.
6. **Corpus is the author's own files.** Ten PDFs from one workspace, not a held-out or
   adversarial corpus. Two are versions of the same document (the source of the v1 ties
   and the v2 near-duplicates).
7. **Engine self-check not re-run here.** The STS-B hash match is the build run's
   self-check, reported by that run; this report's independent re-verification covers the
   embedding arrays, the similarity matrices, the margins, the twins, the set-rule
   counts, and the NL counts.

## 9. What v1.0 (10,000-run) must do

- Use corpus v2 (693 chunks, deduplicated) and the order-independent top-1 SET rule,
  with the permutation test (fixed recorded seed) as part of the spec.
- 10,000 independent repetitions, sharded across zones/hosts, recording the embedding
  array sha256 (and retrieval ranking) per run; verdict = count of identical vs
  differing runs.
- Restate claim (C) in terms of the top-1 set-margin distribution.

## 10. Reproduce

See `README.md` for exact commands, versions, and file hashes. Re-running the v2
commands on the same inputs must reproduce the chunk table v2 and the embedding array
hash `e3e0215c3f192fc8ae54c1fdb7cdb17d39966276cce9b528eef5880bdf0efb13`. Re-running the
v1 commands (no `--prefix`) must reproduce the v1 hash
`e3fe5b6d1ca5cd98c467b0c48b09e7abf9d4d33463d752dc0438f1722e48199d`.
