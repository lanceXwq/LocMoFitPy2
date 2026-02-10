import jax.numpy as jnp
from jax import Array


def rotmat(theta: Array, phi: Array) -> Array:
    sθ, cθ = jnp.sin(theta), jnp.cos(theta)
    sϕ, cϕ = jnp.sin(phi), jnp.cos(phi)
    one = jnp.ones((), dtype=theta.dtype)
    t = one - cθ
    # Reuse products
    t_cϕ = t * cϕ
    t_sϕ = t * sϕ
    r00 = one - t_cϕ * cϕ
    r01 = -t_sϕ * cϕ
    r02 = sθ * cϕ
    r10 = r01
    r11 = one - t_sϕ * sϕ
    r12 = sθ * sϕ
    r20 = -sθ * cϕ
    r21 = -sθ * sϕ
    r22 = cθ
    return jnp.stack([r00, r01, r02, r10, r11, r12, r20, r21, r22]).reshape(3, 3)


def unit_sphere_to_cap(
    sphere_pts: Array, half_angle: Array, eps: float = 1e-12
) -> Array:
    # xy and z views (no copies)
    xy = sphere_pts[:, :2]  # (N,2)
    z = sphere_pts[:, 2]  # (N,)

    ca = jnp.cos(half_angle)
    a = (1.0 + ca) * 0.5
    b = (1.0 - ca) * 0.5

    z_cap = b * z + a

    r1 = jnp.linalg.norm(xy, axis=1)
    r2 = jnp.sqrt(jnp.maximum(0.0, 1.0 - z_cap * z_cap))
    s = r2 / jnp.maximum(r1, eps)

    xy_cap = xy * s[:, None]
    return jnp.concatenate([xy_cap, z_cap[:, None]], axis=1)
