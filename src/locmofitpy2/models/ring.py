import math
from typing import Any, Mapping

import equinox as eqx
import jax.numpy as jnp
from jax import Array, device_get

from ..transformations import rotmat
from .unit_models import unit_ring
from .utils import to_py_or_np


class Ring(eqx.Module):
    x: Array
    y: Array
    z: Array  # center poisition
    r: Array  # radius r = 1/c
    theta: Array  # θ: polar angle
    phi: Array  # ϕ: azimuth angle

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
        return ("x", "y", "z", "r", "theta", "phi")

    def parameter_vector(self):
        return jnp.array([self.x, self.y, self.z, self.r, self.theta, self.phi])

    def parameter_dict(self) -> dict[str, Any]:
        d = {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "r": self.r,
            "theta": self.theta,
            "phi": self.phi,
        }

        d_host = device_get(d)

        return {k: to_py_or_np(v) for k, v in d_host.items()}
