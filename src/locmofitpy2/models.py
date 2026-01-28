from .basic_shapes import SphericalCap
from .registry import register_model

register_model("SphericalCap", SphericalCap, aliases=("spcap", "sphericalcap"))
