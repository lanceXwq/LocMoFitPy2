import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..transformations import rotmat, unit_sphere_to_cap
from .unit_models import unit_sphere


class SphericalCap(eqx.Module):
    # Trainable parameters (0-dim arrays)
    x: Array
    y: Array
    z: Array
    c: Array  # curvature; radius r = 1/c
    alpha: Array  # α: cap half-angle
    theta: Array  # θ: rotation angle
    phi: Array  # ϕ: rotation-axis azimuth (per your rot_trans convention)

    # Non-trainable buffer
    unit_sphere_pts: Array

    @classmethod
    def init(cls, *, x, y, z, c, alpha, theta, phi, dtype, spacing) -> "SphericalCap":
        npoints = int(np.ceil(4 * np.pi / (c * c) / (spacing * spacing)))
        return cls(
            x=jnp.array(x, dtype=dtype),
            y=jnp.array(y, dtype=dtype),
            z=jnp.array(z, dtype=dtype),
            c=jnp.array(c, dtype=dtype),
            alpha=jnp.array(alpha, dtype=dtype),
            theta=jnp.array(theta, dtype=dtype),
            phi=jnp.array(phi, dtype=dtype),
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

    def parameter_dict(self):
        d = {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "c": self.c,
            "alpha": self.alpha,
            "theta": self.theta,
            "phi": self.phi,
        }
        out = {}
        for k, v in d.items():
            v_host = jax.device_get(v)  # bring to host (CPU)
            out[k] = float(np.asarray(v_host))  # robust scalar conversion
        return out
