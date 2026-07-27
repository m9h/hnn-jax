"""The HNN test-oracle contract for hnn-jax.

Three oracle roles (see README):
  1. FORWARD    — hnn-jax(theta) vs hnn_core reference dipole (needs an oracle shard from NSG).
  2. GRADIENT   — hnn-jax autodiff d(dipole)/d(theta) vs finite-difference of the reference (later).
  3. PROPERTY   — metamorphic/invariant checks that need NO oracle (run in CI today).

Property tests go GREEN now (they validate the model runs + behaves sanely). The forward/gradient
tests are RED until an oracle shard is generated on NSG (oracle/data/oracle_shard_*.npz) — NSG's
upload staging is currently down, so those are skipped-with-reason rather than silently passing.
"""
import glob
import os
import pathlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("diffrax")
import jax.numpy as jnp
from hnnjax import simulate, dipole_of, PARAM_NAMES

REPO = pathlib.Path(__file__).resolve().parents[1]
SHARDS = sorted(glob.glob(str(REPO / "oracle" / "data" / "oracle_shard_*.npz")))
FORWARD_CORR_MIN = float(os.environ.get("FORWARD_CORR_MIN", "0.5"))  # ratchet up as fidelity improves

DEFAULT = np.array([2.47, 1.0, 1.0, 1.0])  # canonical-ish theta in PARAM_NAMES order


# ---------- PROPERTY oracle (no reference needed; GREEN today) ----------
def test_runs_and_finite():
    ts, dpl = simulate(DEFAULT)
    assert dpl.shape == ts.shape and dpl.shape[0] > 100
    assert bool(jnp.all(jnp.isfinite(dpl)))


def test_quiescence_with_zero_drive():
    _, dpl = simulate(np.array([2.47, 0.0, 0.0, 0.0]))
    _, base = simulate(np.array([2.47, 0.0, 0.0, 0.0]))
    # no drive -> dipole stays near its resting offset (tiny dynamic range)
    assert float(jnp.max(dpl) - jnp.min(dpl)) < 5.0
    assert bool(jnp.allclose(dpl, base))  # deterministic


def test_drive_gain_monotonicity():
    # stronger proximal-1 drive -> larger early deflection (metamorphic relation)
    rng = [0.5, 1.0, 2.0]
    amp = [float(jnp.max(jnp.abs(simulate(np.array([2.47, g, 0.0, 0.0]))[1]
                                  - simulate(np.array([2.47, 0.0, 0.0, 0.0]))[1]))) for g in rng]
    assert amp[0] < amp[1] < amp[2]


def test_autodiff_is_finite():
    # gradients must at least exist and be finite (correctness vs FD is the GRADIENT oracle below)
    g = jax.grad(lambda th: jnp.sum(dipole_of(th) ** 2))(jnp.asarray(DEFAULT))
    assert g.shape == (len(PARAM_NAMES),) and bool(jnp.all(jnp.isfinite(g)))


# ---------- FORWARD oracle (needs an NSG-generated shard) ----------
@pytest.mark.skipif(not SHARDS, reason="no oracle shard yet (NSG upload staging down) — RED until generated")
def test_forward_matches_oracle():
    d = np.load(SHARDS[0])
    theta, dip_ref, sfreq = d["theta"], d["dipole"], float(d["sfreq"])
    tstop = 1000.0 * (dip_ref.shape[1] - 1) / sfreq          # ms — the oracle's true duration
    t_ref = np.linspace(0.0, tstop, dip_ref.shape[1])         # oracle time axis (ms)
    corrs = []
    for th, ref in zip(theta, dip_ref):
        ts, sim = simulate(th, tstop=tstop)                   # jax on its own grid
        sim_on_ref = np.interp(t_ref, np.asarray(ts), np.asarray(sim))  # resample to common axis
        corrs.append(float(np.corrcoef(sim_on_ref, ref)[0, 1]))
    mean_corr = float(np.nanmean(corrs))
    print(f"\nforward corr vs oracle: mean={mean_corr:.3f} over {len(corrs)} theta "
          f"(tstop={tstop:.0f}ms, sfreq={sfreq:.0f}Hz)")
    assert mean_corr >= FORWARD_CORR_MIN


@pytest.mark.skip(reason="GRADIENT oracle: needs finite-difference Jacobian shard (oracle grads=on)")
def test_gradient_matches_finite_difference():
    ...  # compare jax.jacobian(dipole_of)(theta) vs the FD Jacobian stored in the oracle shard
