import numpy as np

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
        "alpha": float(np.pi / 2),
        "theta": 0.0,
        "phi": 0.0,
    }
}
