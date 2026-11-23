# Methodology

Stubs. Each section is filled in by the work that builds it.

## Question

## Physics model

The regolith is a one-dimensional column with depth-dependent density and temperature-dependent conductivity and heat capacity, following Hayne et al. (2017), *Global regolith thermophysical properties of the Moon from the Diviner Lunar Radiometer Experiment*, JGR Planets 122, 2371–2400 (doi:10.1002/2017JE005387). Code: `src/lunar_thermal/properties.py`.

Equation and table numbers below are those of the openly available arXiv preprint of the paper (arXiv:1711.00977), Appendix A. The published version was not accessible when this was written; if its numbering differs, these citations need updating.

| Property | Formula | Source |
|---|---|---|
| Density | ρ(z) = ρd − (ρd − ρs) e^(−z/H) | Eq. A2 (Eq. 1 in the main text) |
| Contact conductivity | Kc = Kd − (Kd − Ks)(ρd − ρ)/(ρd − ρs) | Eq. A5 |
| Conductivity | K(z, T) = Kc(z) [1 + χ (T / 350 K)³] | Eq. A4 (Eq. A3 with B = Kc χ / 350³) |
| Heat capacity | cp(T) = c0 + c1 T + c2 T² + c3 T³ + c4 T⁴ | Eq. A6 |
| Skin depth (sets the grid) | zs = √(κ P / π), κ = K / (ρ cp) | Eq. A31 |

| Parameter | Default | Source |
|---|---|---|
| H (scale height) | 0.06 m | Table A1, "avg. = 0.06 m" (see the note below) |
| ρs, ρd | 1100, 1800 kg m⁻³ | Table A1 |
| Ks, Kd | 7.4×10⁻⁴, 3.4×10⁻³ W m⁻¹ K⁻¹ | Table A1 |
| χ | 2.7 | Table A1 |
| c0 … c4 | −3.6125, 2.7431, 2.3616×10⁻³, −1.2340×10⁻⁵, 8.9093×10⁻⁹ (J kg⁻¹ K⁻⁽ⁱ⁺¹⁾) | Table A1 (fit to Hemingway et al. 1981 and Ledlow et al. 1992) |
| Emissivity ε | 0.95 | Table A1 |
| Geothermal flux Q | 0.018 W m⁻² | Table A1 (Langseth et al. 1976) |

Where the paper gives more than one value (open questions, not settled here):

- **H.** Table A1 gives an average of 0.06 m, and the paper's standard-model figures (A2, A3) use 0.06 m. Section 4 and Figure 9 give a global (±60°) mean and mode of 0.068 m, and the abstract says "~7 cm". The default is 0.06 m. For the equatorial sanity case, 0.068 m lowers the night minimum by 0.8 K and leaves the noon maximum unchanged.
- **Kd.** The caption of Figure 5 lists Kd = 4.3×10⁻³ W m⁻¹ K⁻¹; Table A1 and the text give 3.4×10⁻³. The default follows Table A1; the caption looks like a transposition.

The heat-capacity fit is stated valid from below 90 K to above 400 K and turns negative below about 1.3 K; values below about 90 K (polar night) are extrapolation. `specific_enthalpy`, the exact integral of cp, is this project's addition. The solver steps it so that energy is conserved exactly while cp varies.

## Ground-truth solver

Code: `src/lunar_thermal/solver.py`; entry point `solve_periodic`. The numerical scheme is this project's own. It solves the equations of Hayne et al. (2017) Appendix A, but does not follow the paper's explicit scheme (their Appendix A2), and `heat1d`'s code was not read. That keeps the triangulation against `heat1d` (deliverable 5) a comparison between independent implementations. Numba compiles the time-stepping kernel; all numerics are implemented here. Written with AI assistance (Claude Code) and checked by the tests listed below.

**Equations.** ρ cp ∂T/∂t = ∂/∂z (K ∂T/∂z) (Eq. A1). Surface: K ∂T/∂z|₀ + F(t) = ε σ T₀⁴ (Eq. A7), where F is the absorbed flux supplied by the caller (albedo and incidence already applied). Bottom: K ∂T/∂z = Q at the base (Eq. A12).

**Contract.** `solve_periodic(absorbed_flux, dt, params=None, grid=None, spinup=None, output_depths=None) -> SolverResult`. The forcing is one period sampled at uniform step `dt`, so the period is `len(absorbed_flux) * dt`, and it wraps around. Between samples the flux is interpolated linearly. The result holds the sample times, surface temperature at each sample, temperatures at the requested depths (interpolated linearly between nodes), the grid, all node temperatures over the final cycle, the end-of-cycle profile, the cycle count, a converged flag, the final cycle-to-cycle change and the projected remaining drift. Output depths are optional; whether to use them is open decision 1.

**Spatial discretization.** Finite volumes. Nodes sit at depths z₀ = 0 < z₁ < … < z_N; each owns the slab halfway to its neighbours (half slabs at both ends), with mass ρ(zᵢ)·wᵢ. The flux between nodes i and i+1 is ½(Kᵢ + Kᵢ₊₁)(Tᵢ₊₁ − Tᵢ)/Δzᵢ, which is second-order accurate and conserves energy by construction. The surface node's half slab carries the radiative balance, and the bottom node's half slab receives Q.

**Grid.** The top layer is zs/m thick, where zs is the skin depth of the surface material at 250 K. Layers then grow by a factor (1 + 1/n) (Hayne et al. Eq. A33) until the column reaches b skin depths of the deep material. Skin depths are taken at 250 K, near the diffusivity minimum, so the top layer is conservatively thin. Using the deep material's skin depth for the bottom matters: it is 1.7 times larger than the surface one for the standard parameters. Defaults m = 20, n = 10, b = 10 give 39 nodes, a 2.0 mm top layer and a 0.73 m column for one lunation. The grid depends on the period, so a longer forcing automatically gets a deeper column.

**Time stepping.** BDF2 on enthalpy: ρw(3h(Tⁿ⁺¹) − 4h(Tⁿ) + h(Tⁿ⁻¹))/(2Δt) = net heat into the slab at tⁿ⁺¹. The very first step of a run uses backward Euler. Each forcing interval is split into ⌈dt / max_step⌉ equal substeps (default max_step 1800 s). Each step is solved with Newton's method on the whole column: the Jacobian is tridiagonal and includes the T-derivatives of K, cp and εσT⁴. It converges to a largest update below 10⁻⁸ K, typically in 2–3 iterations. A Newton failure raises an error rather than returning a result.

Alternatives considered:

- *Explicit (forward Euler), as in the paper.* Simple, but its stability limit is set by the thinnest layer, where accuracy needs thin layers: about 580 s for the 2 mm surface layer at noon. It would also replicate the reference implementation's scheme, weakening the triangulation.
- *Crank–Nicolson.* Second order, but not L-stable, so terrain shadowing (abrupt on/off forcing) would excite undamped oscillations in the stiff surface layer.
- *Backward Euler.* L-stable but first order. BDF2 is also L-stable and second order, at the cost of one extra stored state.
- *Linearising T⁴ once per step instead of iterating Newton.* Cheaper, but its error depends on the step, and the energy balance would close only approximately.

**Spin-up.** The period repeats until two conditions hold. First, the largest change between cycles, over every node and sample time, is below the tolerance (default 0.01 K). Second, the projected remaining drift is below it too. Once fast transients die, the deep column relaxes by a nearly constant ratio r per cycle (about 0.951 per lunation at the equator). A small per-cycle change can therefore hide a large remaining drift, change·r/(1 − r), about 20 times the last change. With a per-cycle criterion alone, runs stopped 0.2–0.4 K short of steady state at depth. When r is stable over three successive estimates and at most 0.98, the solver extrapolates the end-of-cycle state by change·r/(1 − r) (vector Aitken extrapolation). For equatorial forcing this cuts spin-up from 140–190 cycles to 25–50. The ratio cap and the three-estimate stability check guard against a failure seen in testing, where a transient ratio near 1 caused an overshooting jump. An unconverged run returns `converged = False` with its residual and drift, and raises a `RuntimeWarning`.

**Tests** (`tests/test_solver.py`, `tests/test_energy.py`, `tests/test_properties.py`). Each test isolates one part of the solver. Tolerances are stated in each test.

| Test | Isolates | Tolerance | Measured |
|---|---|---|---|
| (a) Constant properties, fixed sinusoidal surface temperature, against T₀ + A e^(−z/d) sin(ωt − z/d) | interior conduction | 0.25 K (A = 100 K), depths d/4 to 3d | 0.125 K |
| (b) No sunlight, constant properties: Tₛ = (Q/εσ)^¼, T = Tₛ + Qz/K | radiative surface and geothermal bottom | 10⁻³ K | 4×10⁻⁵ K |
| (c) No sunlight, Hayne profile, Q = 0.5 W m⁻²: against RK4 integration of K(z,T) dT/dz = Q | depth- and T-dependent conductivity | 0.05 K | 0.007 K |
| (d) Spin-up flags; converged result against a reference converged to 10⁻⁶ K | convergence bookkeeping and honesty | flags exact; 2× tolerance | 0.006 K |
| (e) Halve grid spacing and time step | discretization error | 0.25 K max, 0.05 K rms | 0.15 K, 0.017 K |
| Bottom at 15 instead of 10 skin depths | bottom boundary placement | 0.01 K | 0.0015 K |
| Energy: mean absorbed + Q = mean emitted, from samples | global energy balance | 0.005 W m⁻² | 0.0009 W m⁻² |
| Energy: stored heat returns to start | periodicity | 10⁻⁴ of daily swing | 1.5×10⁻⁵ |
| Energy: one step per sample, net flux = stored-heat rate | discrete conservation | 10⁻⁶ W m⁻² | 2×10⁻⁹ W m⁻² |

Test (c) raises the geothermal flux because at the default Q the column sits near 24–30 K, where the radiative conductivity term is negligible.

**Convergence study** (equatorial forcing, `practice/solver_grid.ipynb`). The error is mostly spatial and largest at sunrise. Against a reference with eight times finer spacing, the maximum surface error is 0.71 K with the paper's suggested m = 10, n = 5 (21 nodes), 0.19 K at the defaults (39 nodes) and 0.046 K at m = 40, n = 20 (74 nodes): second order. Time error at hourly forcing is at most 0.015 K with 1800 s steps.

**Cost.** One equatorial lunation sampled hourly (708 samples), default settings: about 15 ms per cycle and 0.38 s to periodic steady state (25 cycles), single-threaded, on an Intel Core Ultra 9 275HX (Numba 0.68.1, NumPy 2.5.5). The first call in a fresh environment also compiles the kernel (about 5 s, then cached on disk).

**Sanity check (not a validation).** Flat equator, synthetic clipped-cosine forcing at 1 AU with Hayne's angle-dependent albedo (Eq. A8, A0 = 0.12), default parameters. Noon maximum 385.2 K, pre-dawn minimum 93.7 K, midnight 100.3 K. The Diviner equatorial values quoted in Hayne et al. Table A2 (from Vasavada et al. 2012) are 385 K, 95 K and 101 K. Deliverable 5 does the real comparison.

**Limits.** One-dimensional and driven by the absorbed flux it is given: no lateral conduction, scattered light or terrain re-radiation (see Limitations). Below about 90 K the heat capacity is an extrapolation. Depth outputs are linear interpolations between nodes. Midway between nodes this adds error where the profile curves most: up to 1.1 K in the top centimetre (against 0.15 K at the nodes themselves) and 0.3–0.4 K at 2–5 cm, for the equatorial case. If depth outputs are adopted (open decision 1), placing nodes at the requested depths or using higher-order interpolation would remove most of this.


## Solver validation

## Illumination forcing

## Training data and sampling

## Surrogate (FNO v1)

## Physics-informed surrogate (v2, exploratory)

## Baselines and cost comparison

## Evaluation and metrics

## Held-out evaluation

## Limitations
