# Implementation notes

Why LocMoFitPy is implemented the way it is, with the measurements behind each choice. Its features and math are LocMoFit.jl's (`../LocMoFit.jl/NOTES.md`, `../LocMoFit.jl/derivations/derivation.qmd`); how to use and change the code is in `AGENTS.md`.

## The port to LocMoFit.jl's design (2026-10-28)

LocMoFitPy was rebuilt on LocMoFit.jl's design: models that only describe a shape, a flat parameter vector with `ParameterMap` for freezing and ties, composites with component weights, the likelihood with background and prior, the tangent chart and the constant-area cap. Equinox (models as pytrees, freezing by leaf name) and optax (L-BFGS) were no longer used and were dropped; ties in particular do not fit Equinox's partition by name.

**The optimizer is LocMoFit.jl's, transcribed** (`bfgs.py`): Optim.jl's BFGS with LineSearches.jl's `BackTracking` (the same inverse-Hessian update and resets, the same cubic interpolation and step limits, the same stopping rules), in a jitted `lax.while_loop`. optax has no BFGS, and L-BFGS did not converge in LocMoFit.jl on the double ring and the caps. With the same algorithm, every tutorial fit in float64 equals LocMoFit.jl's to about 12 digits, with the same number of iterations (`tests/test_fit.py`), so parity is checked exactly rather than to the optimizer's tolerance.

**optimistix, tried and not adopted** (`benchmarks/speed/optimistix_2026-10-28.txt`). Its BFGS (version 0.1.0, backtracking Armijo search), warm and jitted, on the five tutorial sites. Its built-in termination is Cauchy's (the changes of the parameters and of the loss both below `atol + rtol·|value|`), with no gradient tolerance; a subclass that overrides `terminate` with our rule (largest gradient component ≤ 1e-4) makes the two comparable:

- **Float64, gradient rule:** the same optimum (loss within 1e-7 per localization, parameters within 0.015) in a similar number of steps, and 15–35% faster (ring 6.8 against 10.4 ms, double ring 36 against 53 ms, cap 0.65 against 0.76 s, cap with ring 0.22 against 0.32 s). The steps are not fewer; the evaluations are: optimistix evaluates the loss at each trial point with `jax.linearize` and, when it accepts the point, gets the gradient from that linearization (one backward pass) without evaluating the loss again, while our loop, as Optim.jl, evaluates the loss at the trial point and then the loss and its gradient again once the point is accepted. That is what LocMoFit.jl's `objective` avoids by caching, and our loop can do the same (task 2).
- **Float32:** with the gradient rule optimistix stalls on the ring, the cap and the cap with ring: its line search (halving, no interpolation) finds no decrease the float32 loss resolves, and it runs into the step limit with the largest gradient component at 2e-4 to 4.5e-4; with its Cauchy rule it stops there instead and reports success. Our loop, with LineSearches.jl's interpolating backtracking, reaches the tolerance.
- **Parity:** a different line search takes a different path, so fits would match LocMoFit.jl only to the optimizer's tolerance.

So optimistix would add Equinox and optimistix as dependencies; its Float64 advantage comes from fewer evaluations, which our loop can get too, and it is less reliable in Float32.

## The tolerance

`tol = 1e-4` on the largest gradient component (of the loss per localization) leaves the parameters close to the optimum compared with their statistical error, but not exactly at it: on the cap with a ring, optimistix's tighter fit moved a parameter by 0.53 (float64) and lowered the loss by 2.4e-5 per localization, 0.008 in log-likelihood over the site. The cap's position scatters by 2–11 nm over fits (LocMoFit.jl's accuracy benchmark), so this is far inside it.
