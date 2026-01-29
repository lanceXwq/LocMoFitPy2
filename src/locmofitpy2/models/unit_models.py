import jax.numpy as jnp

GOLDEN_ANGLE = jnp.pi * (3.0 - jnp.sqrt(5.0))  # same common definition


def unit_sphere(
    dtype=jnp.float32,
    npoints: int = 100,
    *,
    gamma=GOLDEN_ANGLE,
    mode: str = "fibonacci",
):
    i = jnp.arange(npoints, dtype=dtype)

    # z in (-1, 1), avoiding endpoints (standard Fibonacci sphere variant)
    z = 1.0 - 2.0 * (i + 0.5) / npoints
    r_xy = jnp.sqrt(jnp.maximum(0.0, 1.0 - z * z))
    phi = i * jnp.asarray(gamma, dtype=dtype)

    x = r_xy * jnp.cos(phi)
    y = r_xy * jnp.sin(phi)

    return jnp.stack([x, y, z], axis=1).astype(dtype)
