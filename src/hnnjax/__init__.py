"""hnn-jax: a differentiable JAX reimplementation of the Human Neocortical Neurosolver forward model,
validated against an HPC-generated hnn_core (NEURON) test oracle."""
from .model import simulate, dipole_of, PARAM_NAMES, BASE

__all__ = ["simulate", "dipole_of", "PARAM_NAMES", "BASE"]
__version__ = "0.0.1"
