"""Differentiable JAX reimplementation of the HNN auditory-ERP forward model (v0 — WIP).

Interface matches oracle/study.yaml: theta = (tc_sync, prox1_gain, dist_gain, prox2_gain) -> dipole(t).
Validated against the hnn_core (NEURON) oracle in oracle/ — the oracle is the arbiter of fidelity.

v0 is deliberately reduced: a single mean pyramidal cell as 2 compartments (soma + apical dendrite),
passive leak + electrotonic coupling + a slow K_m current, driven by *reparameterized* Gaussian
proximal/distal synaptic conductance transients (so gradients flow through drive timing/strength —
notably tc_sync = the proximal drive's timing spread, the protocol's "thalamocortical synchrony").
The dipole readout is the axial dendritic current (∝ Vd - Vs). This captures the ERP's drive-timing
structure and is fully autodiff-able, but omits the spiking network, full HH channels, and the
100-pyramidal-neuron population.

Roadmap (each step checked against the oracle forward + finite-difference gradient):
  v1: population of N cells with per-cell reparameterized drive jitter (vmap); dipole = mean.
  v2: multicompartment cable + full HH (Na/K/Ca/Km) via Diffrax stiff solver.
  v3: layer 2/3 + layer 5 pyramidal + basket interneurons + local connectivity.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import diffrax as dfx

PARAM_NAMES = ("tc_sync", "prox1_gain", "dist_gain", "prox2_gain")

# canonical Jones-2009 ERP drive timing (ms); mirrors oracle/generate_shard.py BASE.
BASE = dict(prox1_mu=26.61, dist1_mu=63.53, dist1_sigma=3.85, prox2_mu=137.12, prox2_sigma=8.33)


def _bump(t, mu, sigma, gain):
    """Differentiable synaptic-conductance transient (normalized Gaussian in time)."""
    return gain * jnp.exp(-0.5 * ((t - mu) / sigma) ** 2)


def _field(t, y, p):
    Vs, Vd, w = y                          # soma V, dendrite V, slow-K gate
    gL, gc, Er, Esyn = 0.1, 0.2, -65.0, 0.0
    gkm, Ek, tauw = 1.0, -90.0, 80.0
    # distal drive -> dendrite; two proximal drives -> soma (prox1 timing spread = tc_sync)
    gd = _bump(t, BASE["dist1_mu"], BASE["dist1_sigma"], p["dist_gain"])
    gp = (_bump(t, BASE["prox1_mu"], p["tc_sync"], p["prox1_gain"])
          + _bump(t, BASE["prox2_mu"], BASE["prox2_sigma"], p["prox2_gain"]))
    dVs = -gL * (Vs - Er) - gc * (Vs - Vd) - gp * (Vs - Esyn) - gkm * w * (Vs - Ek)
    dVd = -gL * (Vd - Er) - gc * (Vd - Vs) - gd * (Vd - Esyn)
    winf = 1.0 / (1.0 + jnp.exp(-(Vs + 35.0) / 10.0))
    dw = (winf - w) / tauw
    return (dVs, dVd, dw)


def simulate(theta, tstop: float = 170.0, dt: float = 0.5):
    """theta: array-like (4,) in PARAM_NAMES order -> (ts, dipole) with grads via diffrax adjoint."""
    theta = jnp.asarray(theta)
    p = {n: theta[i] for i, n in enumerate(PARAM_NAMES)}
    ts = jnp.arange(0.0, tstop, dt)
    sol = dfx.diffeqsolve(
        dfx.ODETerm(_field), dfx.Tsit5(), t0=0.0, t1=tstop, dt0=dt, y0=(-65.0, -65.0, 0.0),
        args=p, saveat=dfx.SaveAt(ts=ts), max_steps=1_000_000,
        adjoint=dfx.RecursiveCheckpointAdjoint())
    Vs, Vd, _ = sol.ys
    # axial dendritic current ∝ current dipole. Sign convention matched to hnn_core's aggregate
    # dipole (verified against the oracle: Vd-Vs is anti-correlated, so use Vs-Vd).
    dipole = Vs - Vd
    return ts, dipole


def dipole_of(theta, **kw):
    """Convenience: theta -> dipole only (for vmap/jacobian against the oracle)."""
    return simulate(theta, **kw)[1]
