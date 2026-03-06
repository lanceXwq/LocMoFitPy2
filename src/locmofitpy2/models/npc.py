from typing import Any, Mapping, Tuple

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from ..transformations import rotmat
from .unit_models import unit_ring
from .utils import sq_hinge


class NuclearPoreComplex(eqx.Module):
    x1: Array
    y1: Array
    z1: Array  # center poisition
    r1: Array  # radius
    theta1: Array
    phi1: Array
    x2: Array  # Δx relative to ring 1
    y2: Array  # Δy relative to ring 1
    z2: Array  # Δz relative to ring 1
    r2: Array  # multiplicative factor relative to r1
    theta2: Array  # orientation relative to ring 1
    phi2: Array
    sigma: Array

    unit_pts: Tuple[Array, Array]

    @classmethod
    def init(cls, *, params: Mapping[str, Any], dtype, spacing) -> "NuclearPoreComplex":
        circ1 = 2 * jnp.pi * params["r1"]
        circ2 = circ1 * params["r2"]
        npts = (int(jnp.ceil(circ1 / spacing)), int(jnp.ceil(circ2 / spacing)))
        return cls(
            x1=jnp.array(params["x1"], dtype=dtype),
            y1=jnp.array(params["y1"], dtype=dtype),
            z1=jnp.array(params["z1"], dtype=dtype),
            r1=jnp.array(params["r1"], dtype=dtype),
            theta1=jnp.array(params["theta1"], dtype=dtype),
            phi1=jnp.array(params["phi1"], dtype=dtype),
            x2=jnp.array(params["x2"], dtype=dtype),
            y2=jnp.array(params["y2"], dtype=dtype),
            z2=jnp.array(params["z2"], dtype=dtype),
            r2=jnp.array(params["r2"], dtype=dtype),
            theta2=jnp.array(params["theta2"], dtype=dtype),
            phi2=jnp.array(params["phi2"], dtype=dtype),
            sigma=jnp.array(params["sigma"], dtype=dtype),
            unit_pts=(
                unit_ring(dtype=dtype, npoints=npts[0]),
                unit_ring(dtype=dtype, npoints=npts[1]),
            ),
        )

    def __call__(self):
        # Rotations
        R2 = rotmat(self.theta2, self.phi2)
        R1 = rotmat(self.theta1, self.phi1)

        # Translations
        t2 = jnp.array([self.x2, self.y2, self.z2])
        t1 = jnp.array([self.x1, self.y1, self.z1])

        # Block 1: scale by r1
        X1 = self.unit_pts[0] * self.r1

        # Block 2: scale by r1*r2, then local transform
        X2 = self.unit_pts[1] * (self.r1 * self.r2)
        X2 = X2 @ R2.T + t2

        # Now assemble once
        X = jnp.concatenate([X1, X2], axis=0)

        # Global transform
        X = X @ R1.T + t1
        return X

    def penalty(self):
        pen_c1 = sq_hinge(jnp.square(1 / self.r1) - 1.0)
        pen_c2 = sq_hinge(jnp.square(1 / self.r2) - 1.0)
        pen_theta1 = sq_hinge(self.theta1 - jnp.pi) + sq_hinge(-self.theta1)
        pen_theta2 = sq_hinge(self.theta2 - jnp.pi) + sq_hinge(-self.theta2)
        return pen_c1 + pen_c2 + pen_theta1 + pen_theta2

    @property
    def dtype(self):
        return self.x1.dtype

    @classmethod
    def trainable_names(cls):
        return (
            "x1",
            "y1",
            "z1",
            "r1",
            "theta1",
            "phi1",
            "x2",
            "y2",
            "z2",
            "r2",
            "theta2",
            "phi2",
            "sigma",
        )

    def parameter_vector(self):
        return jnp.array(
            [
                self.x1,
                self.y1,
                self.z1,
                self.r1,
                self.theta1,
                self.phi1,
                self.x2,
                self.y2,
                self.z2,
                self.r2,
                self.theta2,
                self.phi2,
                self.sigma,
            ]
        )

    @classmethod
    def default_params(cls) -> dict[str, float]:
        keys = cls.trainable_names()
        vals = (0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 0.0, 50.0, 1.0, 0.0, 0.0, 0.0)
        return dict(zip(keys, vals))
