#!/usr/bin/env python3
"""Permutation test for retrieval determinism (claim B).

Falsification test: shuffle the chunk order with a FIXED seed, re-run the
retrieval, and confirm the result is unchanged. Two sub-tests:

  Test 1 — ranking implementation. Take the v2 pass-1 embeddings, permute
  the row order (seed PERM_SEED), recompute the cosine matrix from the
  permuted embeddings, compute the top-1 SETS in permuted coordinates, map
  them back to chunk ids, and compare with the sets computed from the
  unpermuted matrix. Also compare the un-permuted cosine matrix elementwise
  with the v2 similarity matrix (exact float64 equality expected: the dot
  matrix is int64-exact and cosine is elementwise from integers).

  Test 2 — embedding order-independence. The permuted-order embedding pass
  (embed_chunks.py --permute-seed PERM_SEED) embedded the same chunks in
  shuffled order (different batch composition). Compare each chunk's vector
  with its v2 pass-1 vector by chunk_id, and compare the array hash of the
  permuted pass reordered back to corpus order with the pass-1 array hash.

Both sub-tests must pass for claim (B) — retrieval determinism — to pass.
A failure in either is reported plainly; the rule is NOT adjusted until it
passes.

Usage:
  python3 scripts/permutation_test.py --table chunk_table_v2.json \
    --embeddings embeddings_pass1_v2.npz \
    --perm-embeddings embeddings_perm_v2.npz \
    --matrix similarity_matrix_v2.npz --seed 20261118
"""
import argparse
import hashlib
import json
import os

import numpy as np

PROOF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PERM_SEED = 20261118


def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).astype("<i2").tobytes()).hexdigest()


def cosine_from_vectors(V):
    S = V.astype(np.int64) @ V.astype(np.int64).T
    norms2 = np.diag(S).astype(np.float64)
    return S.astype(np.float64) / np.sqrt(np.outer(norms2, norms2))


def top1_sets(C):
    rowmax = C.max(axis=1)
    return [frozenset(np.where(C[i] == rowmax[i])[0].tolist()) for i in range(C.shape[0])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="chunk_table_v2.json")
    ap.add_argument("--embeddings", default="embeddings_pass1_v2.npz")
    ap.add_argument("--perm-embeddings", default="embeddings_perm_v2.npz")
    ap.add_argument("--matrix", default="similarity_matrix_v2.npz")
    ap.add_argument("--seed", type=int, default=PERM_SEED)
    args = ap.parse_args()

    table = json.load(open(os.path.join(PROOF, args.table)))
    ids = [c["chunk_id"] for c in table["chunks"]]
    N = len(ids)
    V = np.load(os.path.join(PROOF, args.embeddings))["vectors"]
    C0 = np.load(os.path.join(PROOF, args.matrix))["cosine"]
    assert V.shape == (N, 384), V.shape
    assert C0.shape == (N, N), C0.shape

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    print("permutation seed: %d (first 10 perm indices: %s)" % (args.seed, perm[:10].tolist()))

    # --- Test 1: ranking implementation under permutation ---
    Vp = V[perm]
    Cp = cosine_from_vectors(Vp)
    sets0 = top1_sets(C0)
    sets_p = top1_sets(Cp)
    # map permuted sets back to original coordinates: permuted row r
    # corresponds to original chunk perm[r]; its set (in permuted coords)
    # maps to { perm[j] for j in set } in original coords.
    sets_back = []
    for r in range(N):
        orig_row = int(perm[r])
        back = frozenset(int(perm[j]) for j in sets_p[r])
        sets_back.append((orig_row, back))
    sets_back_sorted = [s for _, s in sorted(sets_back)]
    test1_sets_identical = all(sets_back_sorted[i] == sets0[i] for i in range(N))
    # un-permute the cosine matrix and compare elementwise
    inv = np.empty(N, dtype=int)
    inv[perm] = np.arange(N)
    C_back = Cp[np.ix_(inv, inv)]
    test1_matrix_exact = bool(np.array_equal(C_back, C0))
    print("Test 1 — ranking under permutation:")
    print("  top-1 sets identical after mapping back: %s" % test1_sets_identical)
    print("  un-permuted cosine matrix exactly equal: %s" % test1_matrix_exact)
    if not test1_sets_identical:
        bad = [ids[i] for i in range(N) if sets_back_sorted[i] != sets0[i]]
        print("  differing rows: %s" % bad[:20])

    # --- Test 2: embedding order-independence ---
    pe = np.load(os.path.join(PROOF, args.perm_embeddings))
    pv = pe["vectors"]
    pids = [str(x) for x in pe["chunk_ids"]]
    pperm = pe["permutation"]
    assert list(pperm) == list(perm), "permuted pass used a different permutation"
    assert int(pe["permute_seed"]) == args.seed, "permuted pass used a different seed"
    by_id_perm = {cid: pv[k] for k, cid in enumerate(pids)}
    V1 = np.load(os.path.join(PROOF, args.embeddings))["vectors"]
    by_id_v1 = {cid: V1[k] for k, cid in enumerate(ids)}
    n_match = sum(1 for cid in ids if np.array_equal(by_id_perm[cid], by_id_v1[cid]))
    # reorder permuted vectors back to corpus order and hash
    Vp_back = np.zeros_like(V1)
    for k, cid in enumerate(pids):
        Vp_back[ids.index(cid)] = pv[k]
    h_perm_back = digest(Vp_back)
    h_v1 = digest(V1)
    test2_vectors_identical = (n_match == N)
    test2_hash_match = (h_perm_back == h_v1)
    print("Test 2 — embedding order-independence:")
    print("  per-chunk vectors identical: %d/%d" % (n_match, N))
    print("  reordered-back array hash == pass-1 hash: %s" % test2_hash_match)
    print("  reordered-back sha256: %s" % h_perm_back)
    print("  pass-1 sha256:         %s" % h_v1)

    verdict = bool(test1_sets_identical and test1_matrix_exact and
                   test2_vectors_identical and test2_hash_match)
    print("PERMUTATION TEST VERDICT: %s" % ("PASS" if verdict else "FAIL"))
    json.dump({"seed": args.seed,
               "n_chunks": N,
               "test1_sets_identical": bool(test1_sets_identical),
               "test1_matrix_exact": bool(test1_matrix_exact),
               "test2_vectors_identical": bool(test2_vectors_identical),
               "test2_vectors_match_count": int(n_match),
               "test2_hash_match": bool(test2_hash_match),
               "test2_reordered_back_sha256": h_perm_back,
               "test2_pass1_sha256": h_v1,
               "verdict": "PASS" if verdict else "FAIL"},
              open(os.path.join(PROOF, "permutation_test.json"), "w"), indent=2)
    print("saved permutation_test.json")


if __name__ == "__main__":
    main()
