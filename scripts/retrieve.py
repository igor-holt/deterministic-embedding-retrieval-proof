#!/usr/bin/env python3
"""Self-retrieval over the chunk embeddings + hand-written NL queries.

SIMILARITY DEFINITION (deterministic by construction):
  - int64 dot matrix S = V @ V.T where V is int16 [N, 384]. Integer
    arithmetic is exact and order-independent, so S is bit-identical on
    any machine given the same V.
  - norms2[i] = S[i, i] (exact int64; each <= 384 * 32767^2 < 2^53, so the
    float64 conversion is exact and sqrt is IEEE correctly-rounded).
  - cosine[i, j] = S[i, j] / sqrt(norms2[i] * norms2[j]), elementwise
    float64. Same integers in -> same float out, on any machine.
  No BLAS float summation is involved in the similarity itself.

RETRIEVAL RULE (v2, order-independent) — the top-1 SET:
  top1_set[i] = { j : cosine[i, j] == max_k cosine[i, k] } (exact float64
  equality). The claim is: the source chunk i is always in top1_set[i].
  The set does not depend on corpus order: permuting the corpus permutes
  the set's labels the same way, it never changes membership. This replaces
  the v1 rule (argmax-first), under which exact ties were broken by corpus
  order — the v1 run's 6 "failures" were ties at cosine exactly 1.0 with
  byte-identical twin chunks. The old argmax rule is still computed and
  recorded as a diagnostic (argmax_top1_*) so the two rules can be compared.

MARGINS: the set margin for row i is max minus the largest cosine strictly
below max (the gap to the best non-tied chunk). A ranking flips under float
noise only if two similarities are within ~1e-15, so a set margin >> 1e-12
means the top-1 set is not a float artefact. The old argsort margin
(top1 - top2) is also recorded for continuity.

NL QUERIES (hand-written, verbatim, 2 per PDF — illustrative only, no
accuracy claim attached; the "intended" PDF is the author's judgment):
  recorded with their top-5 chunk ids and source PDFs.

Options (defaults reproduce the v1 run exactly):
  --table PATH       chunk table JSON (default: chunk_table.json)
  --embeddings PATH  embeddings NPZ (default: embeddings_pass1.npz)
  --prefix STR       output prefix, e.g. v2 (default: "")

Outputs: similarity_matrix[_<prefix>].npz (cosine float64 + int64 dots +
norms2), similarity_matrix[_<prefix>].csv, retrieval_results[_<prefix>].json.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

PROOF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EE_DIR = os.path.join(PROOF, "work", "speedrunpapi", "proofs", "exact-embeddings")
sys.path.insert(0, EE_DIR)
import exact_embed as X  # noqa: E402

NL_QUERIES = [
    # (query text, intended pdf_index)
    ("How can waste heat reveal an engineered technosignature?", 0),
    ("What entropy export channels distinguish artificial from natural signals?", 0),
    ("What cryogenic testbed capabilities does the DOE request for quantum computing?", 1),
    ("How should a quantum-classical HPC testbed be validated before dispatch?", 1),
    ("What are the requirements for cryogenic and quantum HPC testbeds in the DOE request for information?", 2),
    ("Which facilities can host quantum computing testbeds?", 2),
    ("How does the Sovereign Invariance Suite generate revenue?", 3),
    ("What is the monetization model for invariance-based products?", 3),
    ("What is Igor Holt's experience in infrastructure operations and systems architecture?", 4),
    ("Which site reliability roles has the candidate applied for?", 4),
    ("What does the GEM audit cost and what does it score?", 5),
    ("How is a single-project audit priced and what proof of work is cited?", 5),
    ("What are the thirteen revenue streams in the Genesis revenue plan?", 6),
    ("How does Genesis Conductor plan to reach financial liberation?", 6),
    ("What is the verification-before-dispatch principle in the DOE response?", 7),
    ("How does deterministic embedding verification support the quantum testbed proposal?", 7),
    ("How does the Forbes 2026 benchmark rank the candidate?", 8),
    ("What benchmark results support the Forbes 35 Under 35 trajectory?", 8),
    ("How are verifiable quantum circuits compiled and checked?", 9),
    ("What is the scoring rule for the quantum circuit compilation competition?", 9),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="chunk_table.json")
    ap.add_argument("--embeddings", default="embeddings_pass1.npz")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()
    suf = ("_" + args.prefix) if args.prefix else ""

    table = json.load(open(os.path.join(PROOF, args.table)))
    ids = [c["chunk_id"] for c in table["chunks"]]
    pdf_of = [c["pdf_index"] for c in table["chunks"]]
    d = np.load(os.path.join(PROOF, args.embeddings))
    V = d["vectors"].astype(np.int64)
    N = len(ids)
    assert list(d["chunk_ids"]) == ids, "embedding chunk_ids do not match table order"
    t0 = time.time()
    S = V @ V.T
    norms2 = np.diag(S).astype(np.float64)
    denom = np.sqrt(np.outer(norms2, norms2))
    C = S.astype(np.float64) / denom
    print("similarity matrix %s computed in %.1f s" % (C.shape, time.time() - t0))

    # --- top-1 SET rule (order-independent) ---
    rowmax = C.max(axis=1)
    in_set = np.array([i in np.where(C[i] == rowmax[i])[0] for i in range(N)])
    set_sizes = np.array([(C[i] == rowmax[i]).sum() for i in range(N)])
    set_acc = float(in_set.mean())
    # distinct-value (set) margins: max minus largest value strictly below max
    set_margins = []
    for i in range(N):
        below = C[i][C[i] < rowmax[i]]
        set_margins.append(float(rowmax[i] - below.max()) if len(below) else 1.0)
    set_margins = np.array(set_margins)

    # --- diagnostic: old argmax-first rule (v1) ---
    order = np.argsort(-C, axis=1)
    top1 = order[:, 0]
    top1_ok = (top1 == np.arange(N))
    argmax_acc = float(top1_ok.mean())
    in_top5 = np.array([i in order[i, :5] for i in range(N)])
    top5_acc = float(in_top5.mean())
    diag = np.diag(C)
    off = C[~np.eye(N, dtype=bool)]
    margins = np.array([C[i, order[i, 0]] - C[i, order[i, 1]] for i in range(N)])
    set_fails = [ids[i] for i in range(N) if not in_set[i]]
    argmax_fails = [ids[i] for i in range(N) if not top1_ok[i]]
    from collections import Counter
    size_counts = {int(k): int(v) for k, v in Counter(set_sizes.tolist()).items()}
    print("retrieval rule: top-1 SET (order-independent)")
    print("top-1 set accuracy: %.4f (%d/%d)  set sizes: %s" % (set_acc, int(in_set.sum()), N, size_counts))
    print("top-1 set failures: %s" % (set_fails if set_fails else "none"))
    print("diagnostic argmax-first top-1: %.4f (%d/%d)  failures: %s" % (argmax_acc, int(top1_ok.sum()), N, argmax_fails if argmax_fails else "none"))
    print("top-5 containment: %.4f (%d/%d)" % (top5_acc, int(in_top5.sum()), N))
    print("mean diagonal cosine: %.6f" % float(diag.mean()))
    print("mean off-diagonal cosine: %.6f" % float(off.mean()))
    print("min set margin: %.6e ; median set margin: %.6e" % (float(set_margins.min()), float(np.median(set_margins))))
    print("min argsort margin: %.6e ; median argsort margin: %.6e" % (float(margins.min()), float(np.median(margins))))

    # NL queries: embed with a fresh engine, cosine against chunk matrix
    texts = [c["text"] for c in table["chunks"]]
    tok = X.tokenizer()
    eng = X.Exact(X.load_params())
    qvecs = X.embed(eng, tok, [q for q, _ in NL_QUERIES], 32).astype(np.int64)
    Sq = qvecs @ V.T
    nq = np.sqrt((qvecs * qvecs).sum(1).astype(np.float64))
    Cq = Sq.astype(np.float64) / np.outer(nq, np.sqrt(norms2))
    nl = []
    for qi, (q, intended) in enumerate(NL_QUERIES):
        o = np.argsort(-Cq[qi])[:5]
        nl.append({"query": q, "intended_pdf_index": intended,
                   "top5": [{"chunk_id": ids[j], "pdf_index": int(pdf_of[j]),
                             "cosine": float(Cq[qi, j])} for j in o],
                   "top1_pdf_index": int(pdf_of[o[0]]),
                   "top1_is_intended_pdf": bool(pdf_of[o[0]] == intended)})
    n_intended = sum(1 for r in nl if r["top1_is_intended_pdf"])
    print("NL queries: top-1 lands in intended PDF %d/%d" % (n_intended, len(nl)))

    np.savez(os.path.join(PROOF, "similarity_matrix%s.npz" % suf), cosine=C, dots=S,
             norms2=norms2, chunk_ids=np.array(ids))
    with open(os.path.join(PROOF, "similarity_matrix%s.csv" % suf), "w") as f:
        f.write("query_chunk_id," + ",".join(ids) + "\n")
        for i in range(N):
            f.write(ids[i] + "," + ",".join("%.6f" % v for v in C[i]) + "\n")
    json.dump({"n_chunks": N,
               "retrieval_rule": "top-1 SET (order-independent): top1_set[i] = {j : cosine[i,j] == max_k cosine[i,k]}; claim = source chunk i is in top1_set[i]",
               "corpus_version": args.prefix or "v1",
               "top1_set_accuracy": set_acc, "top1_set_correct": int(in_set.sum()),
               "top1_set_sizes": size_counts, "top1_set_failures": set_fails,
               "argmax_top1_accuracy": argmax_acc, "argmax_top1_correct": int(top1_ok.sum()),
               "argmax_top1_failures": argmax_fails,
               "top5_containment": top5_acc, "top5_correct": int(in_top5.sum()),
               "mean_diagonal_cosine": float(diag.mean()),
               "mean_offdiagonal_cosine": float(off.mean()),
               "min_set_margin": float(set_margins.min()),
               "median_set_margin": float(np.median(set_margins)),
               "min_top1_margin": float(margins.min()),
               "median_top1_margin": float(np.median(margins)),
               "nl_queries": nl,
               "nl_top1_intended_pdf": n_intended, "nl_total": len(nl)},
              open(os.path.join(PROOF, "retrieval_results%s.json" % suf), "w"), indent=2)
    print("saved similarity_matrix%s.npz/.csv, retrieval_results%s.json" % (suf, suf))


if __name__ == "__main__":
    main()
