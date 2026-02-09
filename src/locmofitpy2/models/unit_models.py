import jax.numpy as jnp

GOLDEN_ANGLE = jnp.pi * (3.0 - jnp.sqrt(5.0))


def unit_sphere(
    dtype=jnp.float32,
    npoints: int = 100,
):
    dt = dtype
    i = jnp.arange(npoints, dtype=dt)
    n_f = jnp.asarray(npoints, dtype=dt)
    gamma = jnp.asarray(GOLDEN_ANGLE, dt)

    t = (i + 0.5) / n_f
    z = 1.0 - 2.0 * t
    r_xy = jnp.sqrt(jnp.maximum(0.0, 1.0 - z * z))
    phi = i * gamma

    x = r_xy * jnp.cos(phi)
    y = r_xy * jnp.sin(phi)
    return jnp.stack((x, y, z), axis=1)


def unit_ring(
    dtype=jnp.float32,
    npoints: int = 100,
):
    phi = jnp.linspace(0.0, 2.0 * jnp.pi, npoints, endpoint=False, dtype=dtype)
    x = jnp.cos(phi)
    y = jnp.sin(phi)
    z = jnp.zeros_like(x)
    return jnp.stack((x, y, z), axis=1)
