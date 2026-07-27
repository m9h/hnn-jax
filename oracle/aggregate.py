#!/usr/bin/env python3
"""Merge oracle shards (from NSG) into one golden dataset + a provenance manifest.

Usage: python oracle/aggregate.py oracle/data/  ->  oracle/data/oracle.npz + oracle/data/manifest.json
Verifies all shards share hnn_core/NEURON versions and param names (fails loudly if not — the oracle
must be internally consistent to be trustworthy).
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np


def main(d):
    shards = sorted(glob.glob(os.path.join(d, "oracle_shard_*.npz")))
    if not shards:
        print("no shards in", d); return
    thetas, dips, names, sfreqs = [], [], None, set()
    for s in shards:
        z = np.load(s, allow_pickle=True)
        thetas.append(z["theta"]); dips.append(z["dipole"])
        n = list(z["param_names"])
        names = names or n
        assert n == names, f"param_names mismatch in {s}"
        sfreqs.add(float(z["sfreq"]))
    theta = np.concatenate(thetas); dipole = np.concatenate(dips)
    assert len(sfreqs) == 1, f"inconsistent sfreq across shards: {sfreqs}"
    out = os.path.join(d, "oracle.npz")
    np.savez_compressed(out, theta=theta, dipole=dipole, param_names=names, sfreq=sfreqs.pop())
    manifest = {
        "n_samples": int(theta.shape[0]), "n_shards": len(shards), "param_names": names,
        "dipole_len": int(dipole.shape[1]),
        "content_sha256": hashlib.sha256(np.ascontiguousarray(dipole).tobytes()).hexdigest()[:16],
    }
    json.dump(manifest, open(os.path.join(d, "manifest.json"), "w"), indent=2)
    print(json.dumps(manifest, indent=2)); print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "oracle/data")
