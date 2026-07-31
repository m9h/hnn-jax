#!/usr/bin/env python3
"""Generate one shard of the HNN test oracle on NSG (or any CPU node).

The LLNL/Merlin pattern, gateway-friendly: a declarative study (study.yaml) -> a Sobol slice of
theta -> canonical hnn_core (NEURON) simulations fanned across the node's cores (joblib = our
in-allocation "mini-Flux") -> a compact oracle_shard_<start>.npz of (theta, dipole) + a provenance
manifest. Fan many shards to fill the parameter space; aggregate off-NSG.

Deps install to node-local scratch (NSG venv is read-only). Config via cell.json:
  {"shard_start": 0, "shard_n": 8, "sobol_total": 4096}
This first shard is a smoke: it verifies hnn_core/NEURON versions + dipole output + core-parallelism
on NSG, and returns diagnostics even if a sim errors (red-green).
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {}
for p in (os.path.join(HERE, "cell.json"), "cell.json"):
    if os.path.isfile(p):
        CFG = json.load(open(p)); break
SHARD_START = int(CFG.get("shard_start", 0))
SHARD_N = int(CFG.get("shard_n", 8))
SOBOL_TOTAL = int(CFG.get("sobol_total", 4096))
TSTOP, DT = float(CFG.get("tstop_ms", 170.0)), float(CFG.get("dt_ms", 0.025))
LIBS = os.path.join(os.environ.get("TMPDIR", "/tmp"), "nsgkit-pylibs")

RESULT = {"schema": "hnn-oracle/shard/v1", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
          "shard_start": SHARD_START, "shard_n": SHARD_N, "sobol_total": SOBOL_TOTAL}

# theta spec — must mirror oracle/study.yaml (params block). (name, low, high, log?)
PARAMS = [("tc_sync", 1.0, 40.0, False), ("prox1_gain", 0.25, 4.0, True),
          ("dist_gain", 0.25, 4.0, True), ("prox2_gain", 0.25, 4.0, True)]

# canonical Jones-2009 auditory-ERP drive parameters (hnn_core ERP tutorial baseline).
BASE = dict(
    prox1=dict(mu=26.61, sigma=2.47,
               ampa={'L2_basket': 0.08831, 'L2_pyramidal': 0.01525,
                     'L5_basket': 0.19934, 'L5_pyramidal': 0.00865}),
    dist1=dict(mu=63.53, sigma=3.85,
               ampa={'L2_basket': 0.006562, 'L2_pyramidal': 7e-6, 'L5_pyramidal': 0.1423},
               nmda={'L2_basket': 0.019482, 'L2_pyramidal': 0.004317, 'L5_pyramidal': 0.080074}),
    prox2=dict(mu=137.12, sigma=8.33,
               ampa={'L2_basket': 3e-6, 'L2_pyramidal': 1.43884,
                     'L5_basket': 0.008958, 'L5_pyramidal': 0.684013}),
)
DELAYS_P = {'L2_basket': 0.1, 'L2_pyramidal': 0.1, 'L5_basket': 1.0, 'L5_pyramidal': 1.0}
DELAYS_D = {'L2_basket': 0.1, 'L2_pyramidal': 0.1, 'L5_pyramidal': 0.1}


def pip(pkgs):
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--target", LIBS, *pkgs],
                       capture_output=True, text=True, timeout=2400)
    return r.returncode == 0, round(time.time() - t0, 1), r.stderr[-800:]


def ensure_mechanisms(hnn_core):
    """Compile hnn_core's NEURON .mod mechanisms (pip --target install skips the build hook, so the
    .so is missing -> 'No .so file found in hnn_core/mod'). Idempotent."""
    import glob
    import shutil
    mod = os.path.join(os.path.dirname(hnn_core.__file__), "mod")
    have = glob.glob(os.path.join(mod, "**", "*.so"), recursive=True)
    if have:
        return {"status": "precompiled", "so": [os.path.relpath(s, mod) for s in have[:2]]}
    nrniv = shutil.which("nrnivmodl") or next(
        iter(sorted(glob.glob(os.path.join(LIBS, "**", "nrnivmodl"), recursive=True))), None)
    if not nrniv:
        return {"status": "no_nrnivmodl"}
    env = dict(os.environ)
    env["PATH"] = os.path.dirname(nrniv) + os.pathsep + env.get("PATH", "")
    try:
        import neuron
        for cand in (os.path.join(os.path.dirname(neuron.__file__), ".data", "lib"),
                     os.path.join(os.path.dirname(neuron.__file__), "lib")):
            if os.path.isdir(cand):
                env["LD_LIBRARY_PATH"] = cand + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    except Exception:
        pass
    r = subprocess.run([nrniv], cwd=mod, capture_output=True, text=True, timeout=900, env=env)
    return {"status": "compiled", "rc": r.returncode,
            "so": [os.path.relpath(s, mod) for s in
                   glob.glob(os.path.join(mod, "**", "*.so"), recursive=True)[:2]],
            "stderr_tail": r.stderr[-300:]}


def sobol(n_total, dim, seed=0):
    from scipy.stats import qmc
    return qmc.Sobol(d=dim, scramble=True, seed=seed).random(n_total)


def scale(u, lo, hi, log):
    import numpy as np
    if log:
        return float(np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo))))
    return float(lo + u * (hi - lo))


def simulate(theta):
    """One canonical ERP simulation with theta applied; returns (dipole_array, sfreq) or raises."""
    import numpy as np
    from hnn_core import jones_2009_model, simulate_dipole
    p = dict(zip([n for n, *_ in PARAMS], theta))
    net = jones_2009_model()
    b = BASE
    net.add_evoked_drive(
        'evprox1', mu=b['prox1']['mu'], sigma=p['tc_sync'], numspikes=1, location='proximal',
        weights_ampa={k: v * p['prox1_gain'] for k, v in b['prox1']['ampa'].items()},
        synaptic_delays=DELAYS_P, event_seed=1)
    net.add_evoked_drive(
        'evdist1', mu=b['dist1']['mu'], sigma=b['dist1']['sigma'], numspikes=1, location='distal',
        weights_ampa={k: v * p['dist_gain'] for k, v in b['dist1']['ampa'].items()},
        weights_nmda={k: v * p['dist_gain'] for k, v in b['dist1']['nmda'].items()},
        synaptic_delays=DELAYS_D, event_seed=2)
    net.add_evoked_drive(
        'evprox2', mu=b['prox2']['mu'], sigma=b['prox2']['sigma'], numspikes=1, location='proximal',
        weights_ampa={k: v * p['prox2_gain'] for k, v in b['prox2']['ampa'].items()},
        synaptic_delays=DELAYS_P, event_seed=3)
    dpl = simulate_dipole(net, tstop=TSTOP, dt=DT, n_trials=1)[0]
    return np.asarray(dpl.data['agg'], dtype='float32'), float(1000.0 / DT)


def main():
    import shutil
    RESULT["compilers"] = {c: bool(shutil.which(c)) for c in ("gcc", "cc", "clang")}
    # Prefer the tool's PRE-INSTALLED stack (PY_EXPANSE bundles hnn_core + neuron with mechanisms
    # already compiled). Only pip-install to a writable target for what's genuinely missing, and only
    # compile mechanisms if we had to install hnn_core ourselves.
    need = []
    for m, imp in (("hnn_core", "hnn_core"), ("scipy", "scipy"),
                   ("joblib", "joblib"), ("numpy", "numpy")):
        try:
            __import__(imp)
        except Exception:
            need.append(m)
    RESULT["preinstalled_missing"] = need
    if need:
        ok, secs, err = pip(["numpy<2" if m == "numpy" else m for m in need])
        RESULT["pip_ok"], RESULT["pip_seconds"] = ok, secs
        if not ok:
            RESULT["pip_err"] = err; return _write()
        sys.path.insert(0, LIBS)

    import numpy as np
    try:
        import hnn_core
        RESULT["hnn_core_version"] = hnn_core.__version__
        RESULT["hnn_core_path"] = os.path.dirname(hnn_core.__file__)
        try:
            import neuron
            RESULT["neuron_version"] = neuron.__version__
        except Exception as e:
            RESULT["neuron_version"] = repr(e)
        RESULT["n_cores"] = os.cpu_count()
        if "hnn_core" in need:                       # only compile a self-installed hnn_core
            RESULT["mechanisms"] = ensure_mechanisms(hnn_core)
        else:
            RESULT["mechanisms"] = {"status": "preinstalled"}
    except Exception as e:
        RESULT["import_error"] = repr(e); return _write()

    dim = len(PARAMS)
    U = sobol(SOBOL_TOTAL, dim, seed=0)[SHARD_START:SHARD_START + SHARD_N]
    thetas = np.array([[scale(u[i], PARAMS[i][1], PARAMS[i][2], PARAMS[i][3])
                        for i in range(dim)] for u in U], dtype="float64")

    from joblib import Parallel, delayed

    def one(th):
        try:
            dpl, sf = simulate(th)
            return {"ok": True, "dipole": dpl, "sfreq": sf}
        except Exception as e:
            import traceback
            return {"ok": False, "err": repr(e), "trace": traceback.format_exc()[-800:]}

    t0 = time.time()
    res = Parallel(n_jobs=-1, backend="loky")(delayed(one)(th) for th in thetas)
    RESULT["sim_seconds"] = round(time.time() - t0, 1)
    n_ok = sum(r["ok"] for r in res)
    RESULT["n_ok"], RESULT["n_fail"] = int(n_ok), int(len(res) - n_ok)
    if n_ok:
        good = [(th, r["dipole"]) for th, r in zip(thetas, res) if r["ok"]]
        D = np.stack([d for _, d in good])
        TH = np.stack([th for th, _ in good])
        np.savez_compressed(f"oracle_shard_{SHARD_START:06d}.npz",
                            theta=TH, dipole=D, param_names=[n for n, *_ in PARAMS],
                            sfreq=res[[i for i, r in enumerate(res) if r["ok"]][0]]["sfreq"])
        RESULT["dipole_shape"] = list(D.shape)
        RESULT["dipole_sample_stats"] = {"min": float(D.min()), "max": float(D.max()),
                                         "mean": float(D.mean())}
    fail = next((r for r in res if not r["ok"]), None)
    if fail:
        RESULT["first_failure"] = {"err": fail["err"], "trace": fail.get("trace", "")}
    _write()


def _write():
    RESULT["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open("metrics.json", "w") as f:
        json.dump(RESULT, f, indent=2)
    print(json.dumps({k: v for k, v in RESULT.items() if k != "first_failure"}, indent=2))
    if RESULT.get("first_failure"):
        print("FIRST_FAILURE:", json.dumps(RESULT["first_failure"], indent=2))


if __name__ == "__main__":
    main()
