import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .transformations import rotmat, unit_sphere_to_cap
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
    def init(
        cls, key: Array, *, dtype=jnp.float32, npoints: int = 100
    ) -> "SphericalCap":
        k = jax.random.split(key, 8)
        return cls(
            x=jax.random.normal(k[0], (), dtype=dtype),
            y=jax.random.normal(k[1], (), dtype=dtype),
            z=jax.random.normal(k[2], (), dtype=dtype),
            c=jax.random.uniform(k[3], (), dtype=dtype),
            alpha=jax.random.uniform(k[4], (), dtype=dtype),
            theta=jax.random.uniform(k[5], (), dtype=dtype),
            phi=jax.random.uniform(k[6], (), dtype=dtype),
            unit_sphere_pts=unit_sphere(dtype=dtype, npoints=npoints),
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
        return {k: float(jax.device_get(v)) for k, v in d.items()}
