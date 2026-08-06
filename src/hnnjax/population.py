"""v1: population model — N cells, per-cell reparameterized drive jitter, dipole = mean.

v0 (`model.py`) represents each exogenous drive as a single Gaussian *conductance* bump, i.e. a
mean-field approximation of the population. HNN actually gives every pyramidal neuron its own drive
spike time drawn from N(mu, sigma), and each cell's voltage response is **nonlinear** in its input —
so the mean of the responses is not the response to the mean. v1 makes that explicit:

    t_i = mu + sigma * eps_i        eps_i ~ N(0,1), drawn ONCE and held fixed
                                    (reparameterization -> gradients flow through mu AND sigma)

Each cell gets an alpha-function synaptic conductance triggered at its own t_i; the dipole is the
mean over cells. `sigma` for the proximal drive is `tc_sync`, the "thalamocortical synchrony"
parameter the JoVE protocol targets — so this is exactly the parameter whose *population* effect v0
could not represent.
"""
from __future__ import annotations

import diffrax as dfx
import jax
import jax.numpy as jnp

from .model import BASE, PARAM_NAMES

TAU_SYN = 2.0          # ms, alpha-function synaptic time constant


def _alpha(dt, tau=TAU_SYN):
    """Alpha-function synaptic conductance; zero for dt<=0, smooth peak at dt=tau."""
    x = jnp.clip(dt, 0.0, None) / tau
    return x * jnp.exp(1.0 - x) * (dt > 0)


def _field(t, y, p):
    Vs, Vd, w = y                       # per-cell soma V, dendrite V, slow-K gate
    gL, gc, Er, Esyn = 0.1, 0.2, -65.0, 0.0
    gkm, Ek, tauw = 1.0, -90.0, 80.0
    gd = p["dist_gain"] * _alpha(t - p["t_dist"])
    gp = (p["prox1_gain"] * _alpha(t - p["t_prox1"])
          + p["prox2_gain"] * _alpha(t - p["t_prox2"]))
    dVs = -gL * (Vs - Er) - gc * (Vs - Vd) - gp * (Vs - Esyn) - gkm * w * (Vs - Ek)
    dVd = -gL * (Vd - Er) - gc * (Vd - Vs) - gd * (Vd - Esyn)
    winf = 1.0 / (1.0 + jnp.exp(-(Vs + 35.0) / 10.0))
    return (dVs, dVd, (winf - w) / tauw)


def _one_cell(t_prox1, t_dist, t_prox2, theta, ts, tstop, dt):
    p = {"prox1_gain": theta[1], "dist_gain": theta[2], "prox2_gain": theta[3],
         "t_prox1": t_prox1, "t_dist": t_dist, "t_prox2": t_prox2}
    sol = dfx.diffeqsolve(
        dfx.ODETerm(_field), dfx.Tsit5(), t0=0.0, t1=tstop, dt0=dt, y0=(-65.0, -65.0, 0.0),
        args=p, saveat=dfx.SaveAt(ts=ts), max_steps=1_000_000,
        adjoint=dfx.RecursiveCheckpointAdjoint())
    Vs, Vd, _ = sol.ys
    return Vs - Vd                      # per-cell dipole (sign matched to hnn_core)


def make_jitter(key, n_cells=100):
    """Fixed standard-normal draws, one triple per cell (the reparameterization noise)."""
    return jax.random.normal(key, (3, n_cells))


def simulate(theta, eps, tstop: float = 170.0, dt: float = 0.5):
    """theta in PARAM_NAMES order -> (ts, dipole). Differentiable in theta, incl. `tc_sync`."""
    theta = jnp.asarray(theta)
    tc_sync = theta[0]                                   # proximal-drive timing spread
    ts = jnp.arange(0.0, tstop, dt)
    t_p1 = BASE["prox1_mu"] + tc_sync * eps[0]
    t_d = BASE["dist1_mu"] + BASE["dist1_sigma"] * eps[1]
    t_p2 = BASE["prox2_mu"] + BASE["prox2_sigma"] * eps[2]
    per_cell = jax.vmap(lambda a, b, c: _one_cell(a, b, c, theta, ts, tstop, dt))(t_p1, t_d, t_p2)
    return ts, jnp.mean(per_cell, axis=0)


def dipole_of(theta, eps, **kw):
    return simulate(theta, eps, **kw)[1]
