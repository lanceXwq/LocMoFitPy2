# Differences from LocMoFit.jl

LocMoFitPy is the JAX port of LocMoFit.jl (`../LocMoFit.jl`) and must give the same fitted parameters on the same data, `init_params`, `spacing` and `dtype`. This file lists where the two currently differ. Its differences from the original MATLAB LocMoFit are in `../LocMoFit.jl/MATLAB_DIFFERENCES.md`.

Status as of 2026-10-25. New features are developed in LocMoFit.jl first and ported here afterward, so each one appears in the last section until it is ported.

## By design

These follow from the language and the callers (see the parity section of `../AGENTS.md`):

- **Data layout:** `(N, D)` here, `D×N` in LocMoFit.jl.
- **Model names:** `NuclearPoreComplex` here is `DoubleRing` there. Names are looked up case-insensitively here, with aliases (`spcap`, `npc`); LocMoFit.jl needs the exact struct name.
- **Optimizer libraries:** optax here, Optim.jl with Enzyme gradients there, so the iterations differ even for the same method. Parity is therefore checked on fitted parameters, not on per-iteration losses.
- **Return value:** a `dict` of NumPy arrays and Python scalars here (`losses`, `grad_norms`, `model_points`, `parameters`), which MATLAB can convert. LocMoFit.jl returns a `NamedTuple` `(parameters, signal_fraction, weights, components, loss, model, unit_model, model_points, result, history)`; its per-iteration history (losses and parameters) is recorded only with `history=true`, while LocMoFitPy always returns per-iteration losses and gradient norms.

## Current differences to fix

- **Blur.** LocMoFit.jl adds `sigma` to each precision in quadrature, 1/(2(s² + σ²)). LocMoFitPy adds it linearly, (s + σ)⁻²/2 (`utils.py`, `Data.norm_const`).
- **Defaults.** `Ring.r` is 10 here and 50 there.
- **Models.** `SpCapRing` exists only in LocMoFit.jl.
- **Stopping rule.** LocMoFitPy stops when the Euclidean norm of the gradient falls below `tol` (default 1e-6). LocMoFit.jl stops when its largest component falls below `tol` (default 1e-4, reachable in Float32).

## LocMoFit.jl changes not yet ported

- **Composite models with absolute parameters and ties (2026-10-26).** In LocMoFit.jl a composite is `Composite(components...)` of simple models (`Ring`, `SphericalCap`, each defined by its shape in its own frame and placed by a pose `x, y, z, θ, ϕ`):
  - component 1 is placed by its pose in the world, every other component by its pose in component 1's frame;
  - each component's parameters keep their names with its number appended (`x1, r1, …, x2, s2, eta2, …`), and one blur `sigma` is shared;
  - shape parameters are absolute: a ring's radius in nm, a cap's curvature in 1/nm, and a cap is placed by its apex;
  - `tie=(:r2 => :r1,)` makes one parameter take another's value throughout (θ = base + S·u, with the two sharing a column of S);
  - `DoubleRing` and `SpCapRing` are presets of it; each component's points are averaged on their own, with the components weighted (see the next item).

  LocMoFitPy still has the older built-in composite `NuclearPoreComplex`, whose `r2` is a ratio to `r1`, and no `SpCapRing` and no ties. **This is how the Python version will be implemented:** a composite of simple models with the same placement, the same parameter names (in ASCII), absolute shape parameters, and ties.

- **The likelihood with background (2026-10-26).** LocMoFit.jl now fits the posterior of `../LocMoFit.jl/derivations/derivation.qmd`:
  - each localization's density is π·Σₖ wₖ·(the mean over component k's points of the Gaussian) + (1 − π)/V, with the exact Gaussian normalization in 2D and 3D;
  - the signal fraction π = logistic(`logit_prob`) is fitted, with a Beta prior (`signal_prior`, default Beta(5, 2), mode 0.8), and V is the field of view's volume (`fov_volume`; default the localizations' bounding box); `background=false` drops both;
  - a composite's shares are softmax(0, `omega2`, …, `omegaK`), fitted from 0;
  - the squared distances are formed from differences, not expanded, so that Float32 stays accurate at large coordinates;
  - σ starts at 1 nm, since the likelihood is stationary in σ at 0;
  - `run_locmofit` also returns `signal_fraction` and `weights`; the loss is minus the log posterior per localization.

  LocMoFitPy has none of these yet; its normalization constant also differs (it drops the (2π)^(−D/2) and the 1/M). They will be ported the same way.

- **BFGS (2026-10-26).** LocMoFit.jl fits with BFGS and a backtracking line search instead of L-BFGS, as DynMoFit does: with a few tens of parameters its matrix costs nothing, it is insensitive to the parameters' scales, and the fits converge (L-BFGS did not, within 200 iterations, for the double ring, the cap and the cap with ring). LocMoFitPy still uses optax's L-BFGS; it will move to BFGS too (`jax.scipy.optimize.minimize(method="BFGS")`, or its own loop if the per-iteration history must stay).

- **Orientation by a tangent chart, and the constant-area cap (2026-10-26).** In LocMoFit.jl a model's axis is fitted as two tangent-chart coordinates `vx`, `vy` around its starting axis, which `init_params` gives as `theta`, `phi` (`thetak`, `phik` for component k); `theta` and `phi` are no longer parameters, so there is no pole and no angle penalty (`../LocMoFit.jl/derivations/derivation.qmd`). `SphericalCap` is the constant-area cap: scale `s` (area 4πs²) and `eta` with κ′ = tanh(`eta`), its point count from its own area (4–6 times fewer points than before); its only penalty is s ≥ 1 nm. `run_locmofit` also returns `components` (each component's world position, axis and shape) and `unit_model` (the unit points and chart frames) instead of `unit_points`. LocMoFitPy still fits `theta`, `phi` and the cap's `c`, `alpha`; it will be ported the same way.
