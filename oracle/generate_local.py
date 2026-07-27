#!/usr/bin/env python3
"""Generate a small oracle shard locally (dev/CI), reusing the same sim core as the NSG shard.

For scale, use generate_shard.py on NSG; this is for unblocking the diff-test loop off-gateway
(e.g. during an NSG outage). Run with a py<=3.12 env that has hnn_core:
    /tmp/hnn_venv/bin/python oracle/generate_local.py [N]
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_shard import PARAMS, simulate, sobol, scale  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
dim = len(PARAMS)
U = sobol(4096, dim, seed=0)[:N]
thetas = np.array([[scale(u[i], PARAMS[i][1], PARAMS[i][2], PARAMS[i][3]) for i in range(dim)]
                   for u in U], dtype="float64")

dips, ths, sf = [], [], None
t0 = time.time()
for k, th in enumerate(thetas):
    try:
        d, sf = simulate(th)
        dips.append(d); ths.append(th)
        print(f"  sim {k+1}/{N} ok  len={len(d)}")
    except Exception as e:
        print(f"  sim {k+1}/{N} FAIL {e!r}")
        raise

out = os.path.join(os.path.dirname(__file__), "data", "oracle_shard_000000.npz")
os.makedirs(os.path.dirname(out), exist_ok=True)
np.savez_compressed(out, theta=np.stack(ths), dipole=np.stack(dips),
                    param_names=[n for n, *_ in PARAMS], sfreq=sf)
import hnn_core, neuron  # noqa: E402
json.dump({"source": "local", "hnn_core_version": hnn_core.__version__,
           "neuron_version": neuron.__version__, "n": len(ths),
           "dipole_len": int(np.stack(dips).shape[1]), "sfreq": sf,
           "seconds": round(time.time() - t0, 1)},
          open(os.path.join(os.path.dirname(out), "manifest.json"), "w"), indent=2)
print(f"saved {out}  shape={np.stack(dips).shape}  in {time.time()-t0:.1f}s")
