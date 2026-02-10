import jax.numpy as jnp

from .npc import NuclearPoreComplex
from .ring import Ring
from .spcap import SphericalCap


def default_params(model_cls: type) -> dict[str, float]:
    try:
        return dict(_INITIALPARAMS[model_cls])  # copy so callers can mutate
    except KeyError:
        raise TypeError(f"Unsupported model type: {model_cls}")


_INITIALPARAMS: dict[type, dict[str, float]] = {
    SphericalCap: {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "c": 0.02,
        "alpha": float(jnp.pi / 2),
        "theta": 0.0,
        "phi": 0.0,
    },
    Ring: {
        "x": 0.0,
        "y": 0.0,
        "z": 0.0,
        "r": 10.0,
        "theta": 0.0,
        "phi": 0.0,
    },
    NuclearPoreComplex: {
        "x1": 0.0,
        "y1": 0.0,
        "z1": 0.0,
        "r1": 50.0,
        "theta1": 0.0,
        "phi1": 0.0,
        "x2": 0.0,
        "y2": 0.0,
        "z2": 50.0,
        "r2": 1.0,
        "theta2": 0.0,
        "phi2": 0.0,
    },
}
