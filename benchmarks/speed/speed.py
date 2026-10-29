"""Speed of LocMoFitPy's fits, as LocMoFit.jl's benchmarks/speed/speed.jl: for every model on its tutorial site (notebooks/data), from the moved start of the tests, in float64 and float32, the loss and its gradient (jitted) and a complete fit, both a whole `run_locmofit` call (which compiles) and the jitted fit alone. Run from the LocMoFitPy folder:
    pixi run -e dev python benchmarks/speed/speed.py
"""

import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).parents[2] / "tests"))
from common import moved, tutorial  # noqa: E402

import locmofitpy2 as lm  # noqa: E402
from locmofitpy2.bfgs import bfgs  # noqa: E402

CASES = (("ring", "Ring", 3), ("ring2D", "Ring", 2), ("double_ring", "DoubleRing", 3), ("spcap", "SphericalCap", 3), ("spcap_ring", "SpCapRing", 3))


def best(f, n=5):
    f()
    times = []
    for _ in range(n):
        t = time.perf_counter()
        jax.block_until_ready(f())
        times.append(time.perf_counter() - t)
    return min(times)


print(f"jax {jax.__version__} on {jax.devices()[0].platform}")
print(f"{'case':12s} {'dtype':8s} {'N':>5s} {'M':>6s} {'value ms':>9s} {'grad ms':>9s} {'v ns/NM':>8s} {'g ns/NM':>8s} {'call s':>7s} {'fit s':>7s} {'iter':>5s}")
for case, model, D in CASES:
    locs, precs, truth, _, _ = tutorial(case, D)
    init, freeze = moved(truth), ()
    if D == 2:
        init["z"], freeze = truth["z"], ("z",)
    for dtype in ("float64", "float32"):
        kw = dict(init_params=init, freeze=freeze, dtype=dtype, fov_volume=300.0**D)
        t = time.perf_counter()
        r = lm.run_locmofit(model, locs, precs, **kw)
        call = time.perf_counter() - t
        p, u = lm.fit_problem(model, locs, precs, **kw)
        loss = lm.make_loss(p)
        value, grad = jax.jit(loss), jax.jit(jax.grad(loss))
        tv, tg = best(lambda: value(u)), best(lambda: grad(u))
        fit = jax.jit(lambda u: bfgs(loss, u))
        tf = best(lambda: fit(u).x, n=3)
        N, M = len(locs), sum(len(rg) for rg in p.ranges)
        print(f"{case:12s} {dtype:8s} {N:5d} {M:6d} {1e3 * tv:9.3g} {1e3 * tg:9.3g} {1e9 * tv / (N * M):8.2f} {1e9 * tg / (N * M):8.2f} {call:7.3g} {tf:7.3g} {r['iterations']:5d}")
