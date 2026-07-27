# hnn-jax

**A differentiable JAX reimplementation of the Human Neocortical Neurosolver (HNN) forward model,
validated against an HPC-generated `hnn_core` (NEURON) test oracle.**

HNN links localized EEG/MEG biomarkers (e.g. auditory ERP P1/N1/P2) to their cell- and circuit-level
generators via a biophysical neocortical-column model. The reference implementation (`hnn_core`, on
NEURON) is CPU-bound and **non-differentiable**, so parameter fitting relies on black-box search
(CMA-ES) and simulation-based inference. This project builds a **differentiable, GPU-vectorized**
forward model in JAX (Diffrax) — unlocking gradient-based fitting, flow/score-based SBI, and `vmap`
over thousands of parameter sets — and holds it honest against a comprehensive reference oracle.

## The test oracle (the point of this repo)

For a *differentiable* reimplementation the oracle has **three jobs**:

1. **Forward oracle** — `hnn_core(θ) → reference dipole(t)`. hnn-jax must match within tolerance
   *across the parameter space*, not on a few hand-picked cases.
2. **Gradient oracle** — since `hnn_core` isn't differentiable, ground-truth Jacobians `∂dipole/∂θ`
   are **finite differences of the reference**; hnn-jax's autodiff gradients are checked against them.
   (Most JAX reimplementations skip this and ship wrong gradients.)
3. **Property / metamorphic oracle** — invariants needing no reference (zero-drive quiescence, drive
   superposition, gain monotonicity, finite autodiff). These run in CI today.

"Best HPC version" = generate (1)+(2) at **Sobol-dense scale** on NSG/Expanse, versioned and
reproducible — a golden dataset + a `pytest` differential-test harness others can rerun.

## Architecture (LLNL cognitive-simulation pattern, gateway-adapted)

LLNL's stack for "massive sim ensemble → ML" is **Merlin** (ensembles) + **Maestro** (declarative
study spec) + **Flux** (in-allocation hierarchical scheduling). NSG is a locked gateway (no broker,
no Flux), so we adopt the *pattern*, not the software:

```
oracle/study.yaml     # Maestro/Merlin-compatible declarative parameter study (θ ranges, sampling, provenance)
oracle/generate_shard.py  # NSG entrypoint: Sobol slice of θ -> hnn_core sims fanned across the node's
                          #   cores (joblib = in-allocation "mini-Flux") -> compact shard .npz + manifest
oracle/aggregate.py   # merge shards -> golden oracle.npz + provenance manifest (version-consistent)
src/hnnjax/model.py   # the differentiable JAX forward model (Diffrax); θ -> dipole(t), autodiff-able
tests/test_oracle_contract.py  # the 3-role oracle contract (property GREEN now; forward/gradient vs oracle)
```

The same `study.yaml` runs under **real Merlin+Flux on the DGX/Legion** for local generation, and as
an in-allocation joblib ensemble on **NSG** for free Expanse CPU scale — one spec, two backends.

## Status

- **Model:** v0 — reduced 2-compartment pyramidal cell + slow K_m, reparameterized Gaussian drives
  (so gradients flow through drive timing; `tc_sync` = the "thalamocortical synchrony" the protocol
  targets). Runs, autodiff-able, property tests pass. Roadmap in `model.py` → population → full
  multicompartment HH → L2/3+L5 network.
- **Oracle:** shard generator + study spec + aggregator + contract written. First NSG validation
  shard is **blocked by an NSG-side outage** (upload staging returns "error creating tmp file" for
  *all* tools right now); queued to retry. It doubles as the `hnn_core`-version/parallelism check.

Fidelity is tracked, not asserted-perfect: the forward contract compares waveform correlation with a
threshold we ratchet up (`FORWARD_CORR_MIN`) as the model matures against the oracle.

MIT licensed. Related: [`nsg-agent-kit`](https://github.com/m9h/nsg-agent-kit) (the NSG run harness).
