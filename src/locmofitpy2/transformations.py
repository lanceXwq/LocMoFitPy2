import jax.numpy as jnp
from jax import Array


def rotmat(theta: Array, phi: Array) -> Array:
    """
    Args:
        theta: scalar (0-dim) polar angle
        phi:   scalar (0-dim) azimuthal angle

    Returns:
        R: (3, 3) rotation matrix
    """
    s_theta, c_theta = jnp.sin(theta), jnp.cos(theta)
    s_phi, c_phi = jnp.sin(phi), jnp.cos(phi)
    one = jnp.array(1.0, dtype=theta.dtype)
    t = one - c_theta

    return jnp.stack(
        [
            jnp.stack([one - t * c_phi * c_phi, -t * s_phi * c_phi, s_theta * c_phi]),
            jnp.stack([-t * s_phi * c_phi, one - t * s_phi * s_phi, s_theta * s_phi]),
            jnp.stack([-s_theta * c_phi, -s_theta * s_phi, c_theta]),
        ],
        axis=0,
    )


def unit_sphere_to_cap(
    sphere_pts: Array, half_angle: Array, eps: float = 1e-12
) -> Array:
    """
    Args:
        sphere_pts: (N, 3) unit-sphere points.
        half_angle: 0-dim JAX array scalar.
        eps: guard against division by zero when r1 == 0 (points on z-axis).

    Returns:
        cap_pts: (N, 3) points mapped to a spherical cap about +z.
    """
    x = sphere_pts[:, 0]
    y = sphere_pts[:, 1]
    z = sphere_pts[:, 2]

    ca = jnp.cos(half_angle)
    a = (1.0 + ca) / 2.0
    b = (1.0 - ca) / 2.0

    z_cap = b * z + a

    r1 = jnp.sqrt(x * x + y * y)
    r2 = jnp.sqrt(jnp.maximum(0.0, 1.0 - z_cap * z_cap))
    s = r2 / jnp.maximum(r1, eps)

    x_cap = x * s
    y_cap = y * s

    return jnp.stack([x_cap, y_cap, z_cap], axis=1)
