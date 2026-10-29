"""The cap fit of the tutorial data on a CUDA 12 GPU (`pixi run -e cuda12dev python -m pytest tests/test_fit_cuda12.py`); skipped without one."""

import jax
import pytest
from common import fit_on_device, julia_reference

pytestmark = pytest.mark.skipif(
    not any(d.platform == "gpu" for d in jax.devices()), reason="no GPU"
)


def test_spcap_fit_cuda12():
    r = fit_on_device("spcap", "SphericalCap", 3, jax.devices("gpu")[0])
    assert r["retcode"] == "Success"
    assert r["loss"] == pytest.approx(
        julia_reference("spcap")["loss"], rel=1e-8
    )
