import math
from typing import Any, Mapping

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..transformations import rotmat
from .unit_models import unit_ring


class Ring(eqx.Module):
    x: Array
    y: Array
    z: Array  # center poisition
    r: Array  # radius r = 1/c
    theta: Array  # θ: polar angle
    phi: Array  # ϕ: azimuth angle
    sigma: Array

    unit_pts: Array

    @classmethod
    def init(cls, *, params: Mapping[str, Any], dtype, spacing) -> "Ring":
        npoints = math.ceil(2 * math.pi * float(params["r"]) / float(spacing))
        return cls(
            x=jnp.array(params["x"], dtype=dtype),
            y=jnp.array(params["y"], dtype=dtype),
            z=jnp.array(params["z"], dtype=dtype),
            r=jnp.array(params["r"], dtype=dtype),
            theta=jnp.array(params["theta"], dtype=dtype),
            phi=jnp.array(params["phi"], dtype=dtype),
            sigma=jnp.array(params["sigma"], dtype=dtype),
            unit_pts=unit_ring(dtype=dtype, npoints=npoints),
        )

    def __call__(self):
        X = self.unit_pts * self.r
        R = rotmat(self.theta, self.phi)
        t = jnp.array([self.x, self.y, self.z])
        return (X @ R.T) + t

    @property
    def dtype(self):
        return self.x.dtype

    @classmethod
    def trainable_names(cls):
        return ("x", "y", "z", "r", "theta", "phi", "sigma")

    def parameter_vector(self):
        return jnp.array(
            [self.x, self.y, self.z, self.r, self.theta, self.phi, self.sigma]
        )

    @classmethod
    def default_params(cls) -> dict[str, float]:
        keys = cls.trainable_names()
        vals = (0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0)
        return dict(zip(keys, vals))
