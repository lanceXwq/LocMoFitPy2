# AGENTS.md

Guidance for AI coding agents working in this repository. Workspace-wide rules (data layout, units, **parity with LocMoFit.jl**, working with the owner) are in `../AGENTS.md`; read that first.

## What this is

**LocMoFitPy** is the JAX port of LocMoFit.jl (`../LocMoFit.jl`). It fits a static parametric geometric model to one site's localizations by maximum likelihood and **should give the same results as LocMoFit.jl**. It is meant to be called from MATLAB (SMAP, via `pyenv`; see `run_locmofit.m`) and from Python (SMAPpy). Python is capped at the newest version the **latest MATLAB release** supports through `pyenv` (R2026b → 3.14, so `<3.15`). Raise the cap when a new MATLAB release supports a newer Python. Once the lab moves to SMAPpy, follow SMAPpy's Python version instead. Everything else (JAX, Equinox, Optax) tracks the latest stable releases (Compatibility in `../AGENTS.md`).

Names differ by layer, so do not "fix" any of them without asking:

- folder: `LocMoFitPy`
- GitHub repo: `lanceXwq/LocMoFitPy2`
- distribution: `LocMoFitPy2`
- import: `locmofitpy2`

Stack: `jax`, `equinox` (models as pytrees), `optax` (L-BFGS). Environments are managed by **pixi** (`pyproject.toml` → `[tool.pixi.*]`).

## Commands

Run from the repo root; the tests read CSVs by relative path.

```sh
pixi run -e dev python -m pytest tests -k "not cuda12"   # CPU tests (currently FAIL — see Testing)
pixi run -e dev ruff check src tests
pixi run -e dev mypy src
pixi run -e cuda12dev python -m pytest tests/test_spcap_fit_cuda12.py   # Linux + CUDA 12 only
```

Use `python -m pytest`, not the `pytest` entry point. Console-script shebangs in `.pixi/envs` break whenever the folder is moved.

## Layout

| Path | Contents |
| --- | --- |
| `src/locmofitpy2/__init__.py` | Public API: `run_locmofit`, `make_loss`, `fit_lbfgs`, `get_model_cls`, `list_models`, `partition_with_freeze` |
| `src/locmofitpy2/pipeline.py` | `run_locmofit` (the entry point MATLAB calls) |
| `src/locmofitpy2/models/__init__.py` | Imports the models and **registers** them (canonical name + aliases) |
| `src/locmofitpy2/models/registry.py` | `register_model`, `get_model_cls` (case-insensitive), `list_models` |
| `src/locmofitpy2/models/ring.py`, `spcap.py`, `npc.py` | `Ring`, `SphericalCap`, `NuclearPoreComplex` (= LocMoFit.jl's `DoubleRing`) |
| `src/locmofitpy2/models/unit_models.py` | `unit_sphere` (Fibonacci), `unit_ring` |
| `src/locmofitpy2/models/utils.py` | `apply_init`, `parameter_dict`, `sq_hinge` |
| `src/locmofitpy2/models/models.py` | `set_params` (currently unused) |
| `src/locmofitpy2/transformations.py` | `rotmat(theta, phi)`, `unit_sphere_to_cap` |
| `src/locmofitpy2/loss.py` | `pairwise_sqdist`, `negative_log_likelihood`, `make_loss` |
| `src/locmofitpy2/optim.py` | `fit_lbfgs`: `optax.lbfgs` + zoom line search inside a jitted `lax.while_loop` |
| `src/locmofitpy2/utils.py` | `Data(locs, precs).norm_const(blur)`, `partition_with_freeze` |
| `src/locmofitpy2/_profile.py`, `_jaxprofile.py` | Profiling helpers, gitignored (`_*profile.py`) |
| `tests/` | One fit per model, asserting a hard-coded final loss. `example_*_data.csv` are copies of LocMoFit.jl's `simulated_data_*.csv`. |
| `notebooks/` | Owner's fit notebooks. The simulated CSVs in `notebooks/data/` are tutorial data and are committed. `fit spcap ROI4.ipynb` runs on experiment data, which now lives in `../data/`; it still reads `data/260126_*ROI4.csv`, so it fails until its path is updated. |
| `run_locmofit.m` | MATLAB example (`pyenv` OutOfProcess). It hard-codes the env path from before the folder moved (`/Users/lancexwq/Ries Lab/LocMoFitPy2/...`). |
| `benchmarks/`, `profiles/`, `prof.pstats`, `note.md` | Owner's experiments and notes (untracked). Ignore. |

## Model interface (Equinox)

A model is an `eqx.Module` with one scalar `Array` field per parameter, plus `unit_pts` (a tuple of arrays for composite models). It also defines:

- `@classmethod init(cls, *, params, dtype, spacing)`, which builds `unit_pts` **once, from the initial parameters**. The point count (`ceil(circumference/spacing)`, `ceil(area/spacing²)`) is fixed for the whole fit, so the loss value depends on the initial `r`/`c` and `spacing`.
- `@classmethod default_params() -> dict` and `@classmethod trainable_names() -> tuple`. The tuple order must match `parameter_vector()`.
- `__call__(self) -> (M, 3)` model points, `penalty(self) -> scalar`, `parameter_vector(self)`, and a `dtype` property.
- A `register_model(...)` line in `models/__init__.py`. Canonical names should equal LocMoFit.jl's struct names.

Freezing works by leaf **name**: `partition_with_freeze` marks a leaf trainable only if its attribute name is in `trainable_names()` and not in `freeze`, and `unit_pts` is never trainable.

**Mirror every model, default, penalty, or loss change in LocMoFit.jl** (see the parity section in `../AGENTS.md`), or say explicitly that you did not.

## Conventions

- **Data are `(N, D)` (rows are localizations)**, the transpose of LocMoFit.jl. Precisions are per-axis standard deviations of the same shape, in nm. For 2D input, the model points are truncated to their first 2 columns.
- Parameter names are ASCII (`theta`, `phi`, `alpha`, `sigma`); LocMoFit.jl uses `θ`, `ϕ`, `α`, `σ`.
- `dtype="float64"` calls `jax.config.update("jax_enable_x64", True)`. That is **process-global and sticky**: it affects every later JAX call in the same Python process, including a long-lived MATLAB `pyenv` session.
- Everything under `fit_lbfgs` is jitted. The loss must be pure and traceable: no Python branching on traced values, no `print` (use `jax.debug.print`).
- The value `run_locmofit` returns must stay MATLAB-convertible: a `dict` of NumPy arrays and Python scalars (`losses`, `grad_norms`, `model_points`, `parameters`). Never return JAX arrays or Equinox modules.
- Ask before adding dependencies (they go in `pyproject.toml`, and `pixi.lock` changes with them).

## Testing

- The tests assert the final loss to `rtol=1e-6` against hard-coded numbers, and **both CPU tests currently fail**. The expected values (391.17, 96.36) predate the sum→mean change to the NLL and the later σ/penalty changes; current final losses are ~4.49 and ~2.20. Update an expected value only when the owner asks, and only after checking the new number against LocMoFit.jl on the same CSV with the same `init_params`, `spacing` and `dtype`.
- Results are platform-sensitive. The CUDA 12 test is the same fit as the CPU spcap test, yet expects 93.598 instead of 96.357. Expect last-digit drift across JAX/XLA versions. Compare fitted parameters, not loss trajectories, when checking parity.
