#!/usr/bin/env python3
"""Embed chunks with speedrunpapi exact_embed (NumPy Exact engine).

Two INDEPENDENT passes by default: each pass re-loads the model parameters
from disk and builds a fresh Exact engine, then embeds every chunk text in
chunk order (batch 32). Determinism is checked by exact array equality and by
SHA-256 of the int16 embedding bytes (exact_embed.digest).

Model: sentence-transformers/all-MiniLM-L6-v2, pinned revision
1110a243fdf4706b3f48f1d95db1a4f5529b4d41 (pins.json in the cloned repo).
Output: 384 int16 per chunk at scale 2^-15. Engine: NumPy int64 exact
arithmetic; the only float op is double-precision BLAS on integer-valued
arrays (exact per bounds.py).

Options (defaults reproduce the v1 smoke-test run exactly):
  --table PATH        chunk table JSON (default: chunk_table.json)
  --prefix STR        output prefix, e.g. v2 -> embeddings_pass1_v2.npz
                      (default: "" -> embeddings_pass1.npz)
  --hashes PATH       hashes JSON output (default: embedding_hashes.json,
                      or embedding_hashes_<prefix>.json when --prefix given)
  --passes N          number of independent passes (default: 2)
  --permute-seed INT  if given: do ONE pass with the chunk order shuffled
                      by this seed, and save embeddings_perm[_<prefix>].npz
                      with chunk_ids in permuted order + the permutation.
                      Used for the retrieval-determinism permutation test.

Outputs (in the proof dir), e.g. with --prefix v2:
  embeddings_pass1_v2.npz / embeddings_pass2_v2.npz — int16 [N, 384] +
    chunk ids + model/engine metadata. sha256 of each array is printed
    and recorded in embedding_hashes_v2.json.
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

PROOF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EE_DIR = os.path.join(PROOF, "work", "speedrunpapi", "proofs", "exact-embeddings")
sys.path.insert(0, EE_DIR)
import exact_embed as X  # noqa: E402

CACHE = os.path.join(PROOF, "work", "model-cache")
os.environ["EXACT_EMBED_CACHE"] = CACHE


def digest(arr):
    return hashlib.sha256(np.ascontiguousarray(arr).astype("<i2").tobytes()).hexdigest()


def one_pass(texts, batch=32):
    tok = X.tokenizer()
    eng = X.Exact(X.load_params())
    return X.embed(eng, tok, texts, batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="chunk_table.json")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--hashes", default=None)
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--permute-seed", type=int, default=None)
    args = ap.parse_args()

    suf = ("_" + args.prefix) if args.prefix else ""
    table = json.load(open(os.path.join(PROOF, args.table)))
    texts = [c["text"] for c in table["chunks"]]
    ids = [c["chunk_id"] for c in table["chunks"]]
    print("chunks: %d (table %s)" % (len(texts), args.table))

    meta = {"model_repo": X.PINS["model"]["repo"], "model_revision": X.PINS["model"]["revision"],
            "engine": "exact_embed.Exact (NumPy int64)", "batch": 32,
            "speedrunpapi_commit": "8479566cfd3b008f927184d32ac9b6a98deec3a5",
            "chunk_ids": ids}

    if args.permute_seed is not None:
        rng = np.random.default_rng(args.permute_seed)
        perm = rng.permutation(len(texts))
        ptexts = [texts[i] for i in perm]
        pids = [ids[i] for i in perm]
        print("permuted order, seed %d" % args.permute_seed)
        t0 = time.time()
        v = one_pass(ptexts)
        print("perm pass: shape %s dtype %s  %.1f s" % (v.shape, v.dtype, time.time() - t0))
        out = os.path.join(PROOF, "embeddings_perm%s.npz" % suf)
        np.savez(out, vectors=v, chunk_ids=np.array(pids), permutation=perm,
                 permute_seed=np.array(args.permute_seed),
                 **{k: np.array(vv) for k, vv in meta.items() if k != "chunk_ids"})
        print("saved %s (chunk_ids in permuted order)" % out)
        return

    vecs = []
    times = []
    for p in range(args.passes):
        t0 = time.time()
        v = one_pass(texts)
        t1 = time.time()
        print("pass %d: shape %s dtype %s  %.1f s" % (p + 1, v.shape, v.dtype, t1 - t0))
        vecs.append(v)
        times.append(round(t1 - t0, 1))
    hashes = [digest(v) for v in vecs]
    for i, h in enumerate(hashes):
        print("pass %d sha256: %s" % (i + 1, h))
    if len(vecs) == 2:
        print("bit-identical (array_equal): %s" % bool(np.array_equal(vecs[0], vecs[1])))
    for i, v in enumerate(vecs):
        out = os.path.join(PROOF, "embeddings_pass%d%s.npz" % (i + 1, suf))
        np.savez(out, vectors=v,
                 **{k: np.array(vv) for k, vv in meta.items() if k != "chunk_ids"},
                 chunk_ids=np.array(ids))
        print("saved %s" % out)
    hashes_path = args.hashes or ("embedding_hashes%s.json" % suf)
    rec = {"shape": list(vecs[0].shape), "dtype": str(vecs[0].dtype),
           "pass_seconds": times}
    for i, h in enumerate(hashes):
        rec["pass%d_sha256" % (i + 1)] = h
    if len(vecs) == 2:
        rec["bit_identical"] = bool(np.array_equal(vecs[0], vecs[1]))
        rec["pass1_seconds"] = times[0]
        rec["pass2_seconds"] = times[1]
    json.dump(rec, open(os.path.join(PROOF, hashes_path), "w"), indent=2)
    print("saved %s" % hashes_path)


if __name__ == "__main__":
    main()
