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
    vartheta: Array  # ϑ: cap half-angle
    phi0: Array
    # φ₀: present in your Julia params; not used in your forward as written
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
            vartheta=jax.random.uniform(k[4], (), dtype=dtype),
            phi0=jax.random.uniform(k[5], (), dtype=dtype),
            theta=jax.random.uniform(k[6], (), dtype=dtype),
            phi=jax.random.uniform(k[7], (), dtype=dtype),
            unit_sphere_pts=unit_sphere(dtype=dtype, npoints=npoints),
        )

    def __call__(self):
        X = unit_sphere_to_cap(self.unit_sphere_pts, self.vartheta)
        X = (X + jnp.array([0.0, 0.0, -1.0], dtype=X.dtype)) / self.c

        R = rotmat(self.theta, self.phi)
        t = jnp.stack([self.x, self.y, self.z])
        return (X @ R.T) + t
