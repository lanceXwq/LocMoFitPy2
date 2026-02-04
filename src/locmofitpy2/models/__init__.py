from .registry import register_model
from .spcap import SphericalCap

register_model("SphericalCap", SphericalCap, aliases=("spcap", "sphericalcap"))

__all__ = [
    "SphericalCap",
    "register_model",
]
