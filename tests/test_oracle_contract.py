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
FORWARD_CORR_MIN = float(os.environ.get("FORWARD_CORR_MIN", "0.40"))  # ratcheted: v1 achieves 0.43

DEFAULT = np.array([2.47, 1.0, 1.0, 1.0])  # canonical-ish theta in PARAM_NAMES order


def _corr_against_oracle(sim_fn):
    d = np.load(SHARDS[0])
    theta, ref, sf = d["theta"], d["dipole"], float(d["sfreq"])
    tstop = 1000.0 * (ref.shape[1] - 1) / sf
    t_ref = np.linspace(0.0, tstop, ref.shape[1])
    cs = []
    for th, r in zip(theta, ref):
        ts, s = sim_fn(th, tstop)
        cs.append(float(np.corrcoef(np.interp(t_ref, np.asarray(ts), np.asarray(s)), r)[0, 1]))
    return float(np.nanmean(cs))


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
@pytest.mark.skipif(not SHARDS, reason="no oracle shard yet (run oracle/generate_local.py) — RED until generated")
def test_forward_matches_oracle():
    """Forward fidelity of the CURRENT best model (v1 population) against hnn_core.

    v0's mean-field version is kept in `model.py` and compared below; the ratcheted
    FORWARD_CORR_MIN tracks what the current model actually achieves, so a regression goes red.
    """
    import jax
    from hnnjax.population import make_jitter, simulate as sim_v1
    eps = make_jitter(jax.random.PRNGKey(0), 100)
    mean_corr = _corr_against_oracle(lambda th, T: sim_v1(th, eps, tstop=T))
    print(f"\nforward corr vs oracle (v1 population): {mean_corr:.4f}")
    assert mean_corr >= FORWARD_CORR_MIN


@pytest.mark.skip(reason="GRADIENT oracle: needs finite-difference Jacobian shard (oracle grads=on)")
def test_gradient_matches_finite_difference():
    ...  # compare jax.jacobian(dipole_of)(theta) vs the FD Jacobian stored in the oracle shard


# ---------- v1 population model ----------


@pytest.mark.skipif(not SHARDS, reason="needs an oracle shard")
def test_population_model_beats_mean_field():
    """v1 (per-cell jitter, vmapped) must be a real improvement on v0's mean-field bump.

    Averaging cell *responses* is not the response to the *average* drive, because each cell is
    nonlinear in its input. This asserts the gain is real, not noise.
    """
    import jax
    from hnnjax.population import make_jitter, simulate as sim_v1
    eps = make_jitter(jax.random.PRNGKey(0), 100)
    c_v0 = _corr_against_oracle(lambda th, T: simulate(th, tstop=T))
    c_v1 = _corr_against_oracle(lambda th, T: sim_v1(th, eps, tstop=T))
    print(f"\nforward corr vs oracle: v0 mean-field {c_v0:+.4f} -> v1 population {c_v1:+.4f}")
    assert c_v1 > c_v0 + 0.1, f"v1 {c_v1:.3f} not clearly better than v0 {c_v0:.3f}"
    assert c_v1 >= FORWARD_CORR_MIN


@pytest.mark.skipif(not SHARDS, reason="needs an oracle shard")
def test_population_autodiff_through_jitter():
    """Gradients must flow through the reparameterized drive times -- including tc_sync (sigma)."""
    import jax
    import jax.numpy as jnp
    from hnnjax.population import dipole_of, make_jitter
    eps = make_jitter(jax.random.PRNGKey(0), 16)
    g = jax.grad(lambda th: jnp.sum(dipole_of(th, eps) ** 2))(jnp.array([2.47, 1.0, 1.0, 1.0]))
    assert bool(jnp.all(jnp.isfinite(g)))
    assert abs(float(g[0])) > 0, "no gradient w.r.t. tc_sync -- reparameterization broken"
