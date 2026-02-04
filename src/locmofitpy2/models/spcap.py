from typing import Any, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..transformations import rotmat, unit_sphere_to_cap
from .unit_models import unit_sphere
from .utils import to_py_or_np


class SphericalCap(eqx.Module):
    # Trainable parameters (0-dim arrays)
    x: Array
    y: Array
    z: Array
    c: Array  # curvature; radius r = 1/c
    alpha: Array  # α: cap half-angle
    theta: Array  # θ: polar angle
    phi: Array  # ϕ: azimuth angle

    # Non-trainable buffer
    unit_sphere_pts: Array

    @classmethod
    def init(cls, *, params: Mapping[str, Any], dtype, spacing) -> "SphericalCap":
        npoints = int(
            np.ceil(4 * np.pi / (params["c"] * params["c"]) / (spacing * spacing))
        )
        return cls(
            x=jnp.array(params["x"], dtype=dtype),
            y=jnp.array(params["y"], dtype=dtype),
            z=jnp.array(params["z"], dtype=dtype),
            c=jnp.array(params["c"], dtype=dtype),
            alpha=jnp.array(params["alpha"], dtype=dtype),
            theta=jnp.array(params["theta"], dtype=dtype),
            phi=jnp.array(params["phi"], dtype=dtype),
            unit_sphere_pts=unit_sphere(dtype=dtype, npoints=int(npoints)),
        )

    def __call__(self):
        X = unit_sphere_to_cap(self.unit_sphere_pts, self.alpha)
        X = (X + jnp.array([0.0, 0.0, -1.0], dtype=X.dtype)) / self.c
        R = rotmat(self.theta, self.phi)
        t = jnp.stack([self.x, self.y, self.z])
        return (X @ R.T) + t

    @property
    def dtype(self):
        return self.x.dtype

    @classmethod
    def trainable_names(cls):
        return ("x", "y", "z", "c", "alpha", "theta", "phi")

    def parameter_vector(self):
        return jnp.stack(
            [self.x, self.y, self.z, self.c, self.alpha, self.theta, self.phi]
        )

    def parameter_dict(self) -> dict[str, Any]:
        d = {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "c": self.c,
            "alpha": self.alpha,
            "theta": self.theta,
            "phi": self.phi,
        }

        d_host = jax.device_get(d)  # one transfer

        return {k: to_py_or_np(v) for k, v in d_host.items()}
