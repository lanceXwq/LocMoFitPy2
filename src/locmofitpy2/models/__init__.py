from .registry import register_model
from .ring import Ring
from .spcap import SphericalCap

register_model("SphericalCap", SphericalCap, aliases=("spcap", "sphericalcap"))
register_model("Ring", Ring)

__all__ = [
    "Ring",
    "SphericalCap",
    "register_model",
]
