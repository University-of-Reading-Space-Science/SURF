# Native one-dimensional MHD in SURF

Select `solver='mhd'` for HLL fluxes, MC-limited piecewise-linear reconstruction
and second-order SSP RK2, or `solver='mhd-pcm'` for HLL with piecewise-constant
reconstruction and forward Euler. The implementation is Python/Numba in
`surf/surf_mhd.py`; it has no PLUTO, CANS, or other external solver dependency.

## What the existing models do

| Model | Relevant treatment | Consequence for SURF |
| --- | --- | --- |
| [Tao et al. (2005), section 2.1](https://doi.org/10.1029/2004JA010959) | Radial spherical ideal MHD with solar gravity, CANS TVD/MUSCL and a linearized Riemann solver. Time-dependent observations drive the inner boundary; the outer boundary is extrapolated. Their example uses gamma=7/5, Bn=0 and a small fixed inner Br, decreasing as r^-2. | Evolve transverse magnetic pressure and tension, use the fast magnetosonic signal speed, and enforce the radial-flux constraint. |
| [Helio1D, Kieokaew et al. (2024), section 2.2](https://doi.org/10.1051/swsc/2024018) | Couples MULTI-VP at 30 solar radii to the Tao propagation model, with time-varying plasma and tangential field. The published configuration uses inner Br=0.001 nT and Bn=0. | A transverse-field approximation is a useful configuration of a propagation solver. It does not require wrapping the original code. |
| [sunRunner1D, Riley & Ben-Nun (2022), section 2.2](https://doi.org/10.3390/universe8090447) | Uses unmodified PLUTO in spherical 1D, with user-specified density, speed, temperature and transverse magnetic field. Independent boundary pulses represent ICMEs. | Use SURF's existing plasma/CME drivers and supply explicit physical magnetic boundary profiles. The PLUTO-based model found in this review is named sunRunner1D; “helioRunner” was not independently identified. |
| [Pomoell et al. (2015), section 2.1](https://doi.org/10.1051/swsc/2015015) | Relaxes a 1.5D radial background wind with two vector components, gravity and exponential coronal heating. That background feeds a **2D** CME shock model. | Retaining transverse velocity and magnetic tension is useful. A coronal heating/critical-point solver is a separate problem from SURF's heliospheric propagation. This is a verified related Pomoell model, not a claim to have identified every code mentioned in conversation. |

These papers motivate the physical reduction. SURF's HLL implementation is
independent; it does not reproduce any paper's numerical scheme or fitted
boundary conditions. HLL resolves fast shocks but is more diffusive for contact,
Alfven and rotational discontinuities than HLLD. PLM reduces that diffusion.
No observational forecasting skill or agreement with the external codes is
claimed by the numerical tests below.

## Equations and geometry

The model has one spatial coordinate and three physical vector components in
a local radial-tangential-normal (RTN) basis. It represents an expanding radial
flux tube, with cross section proportional to r² and both transverse lengths
proportional to r. Angular gradients and coupling between longitude/latitude
slices are omitted. This is a local radial reduction, not a globally spherically
symmetric vector field or a divergence-free reconstruction of a whole 3D Sun.

In this reduction the solenoidal constraint and radial induction equation give

```text
d(r² Br)/dr = 0,       dBr/dt = 0,
Br(r) = br_boundary (r[0]/r)².
```

Consequently `br` must be a time-independent scalar. Feeding arbitrary measured
Br(t), advecting sector reversals in Br, or rotating a Br longitude map through
this strictly radial solver would violate its equations. Use the transverse
approximation (`br=0`, or a small constant) when appropriate; the existing
`b_boundary`/`b_grid` polarity tracer remains a separate, passive quantity.

Let `Bperp²=Bt²+Bn²` and `P=p+Bperp²/(2 mu0)`. The flow is strictly radial,
with no transverse velocity or transverse momentum variables. Define

```text
E' = p/(gamma-1) + rho*vr²/2 + Bperp²/(2 mu0).
```

The evolved variables are `[rho, rho*vr, E', Bt, Bn]`.
The omitted radial magnetic energy is constant in time; add `Br²/(2 mu0)`
when reporting full magnetic plus plasma energy. With `a` denoting either
transverse component, the equations implemented are

```text
dt rho      + r^-2 dr[r² rho vr] = 0
dt(rho vr)  + r^-2 dr[r² (rho vr² + P)] = 2p/r + rho g
dt E'       + r^-2 dr[r² (E'+P)vr] = rho vr g
dt Ba       + r^-1 dr[r vr Ba] = 0
```

Here `g=0` by default, or `-GM_sun/r²` with `mhd_gravity=True`. The analytic
radial magnetic stress cancels its geometric source before discretization;
this avoids numerical radial acceleration from a force-free Br proportional
to r^-2. Transverse magnetic pressure remains in radial momentum and energy.
The density, radial momentum and energy use shell weights integral(r² dr),
while induction uses integral(r dr). The pressure source uses the shell
average of 1/r.
This preserves the respective finite-volume flux budgets up to roundoff.

Thermal pressure follows SURF hydro's existing convention
`p = rho k_B T / m_p`. In particular, this is **not** the `m_p/2` mean-mass
convention used in the Pomoell paper. Convert temperature/pressure consistently
before comparing codes. Gamma defaults to SURF's 1.5; it is configurable and
must exceed 1. Effective gamma changes shock heating as well as expansion.

The timestep uses `abs(vr)+cf`, where

```text
a² = gamma p/rho,   aperp² = Bperp²/(mu0 rho)
cf² = a² + aperp².
```

The prescribed force-free radial field does not contribute a wave
characteristic because this reduced model has no transverse momentum.

Both cell and boundary signal speeds are checked. The default CFL is 0.35
(accepted range 0 < CFL <= 0.5). Nonphysical reconstructed states revert to
PCM locally. Nonpositive RK stages reject the update and retry with PCM and
a smaller timestep, with a bounded retry count and an explicit error on
failure. Thermal energy is never silently clipped to make a run appear valid.
This fallback is a safeguard, not a mathematical positivity guarantee for
arbitrary initial data. Low-level results report the number of timestep retries.

## Use in SURF

```python
import numpy as np
from astropy import units as u
from surf.surf import SURF

model = SURF(
    v_boundary=np.full(128, 400.0) * u.km/u.s,
    solver='mhd',
    lon_out=0*u.deg,
    r_min=30*u.solRad,
    r_max=215*u.solRad,
    simtime=5*u.day,
    mhd_boundary={'br': 10*u.nT, 'bt': -20*u.nT, 'bn': 0*u.nT},
    mhd_gravity=True,
)
model.solve([])
# Existing outputs: v_grid (km/s), rho_grid (kg/m³), temp_grid (K).
# New outputs: br_grid, bt_grid, bn_grid (tesla).
bt_at_outer_boundary = model.bt_grid[:, -1, 0].to_value(u.nT)
```

`mhd_boundary` keys are `br`, `bt`, `bn`, optionally `time`.
Missing vector components default to **zero**. No magnetic magnitude or Parker
spiral is inferred from speed, polarity, or a CME cone. Supply a nonzero field
explicitly for a magnetized run. Plain numbers are SI (tesla, m/s, seconds);
Astropy quantities are converted and checked.

For the two evolving transverse magnetic components, choose either:

* Scalars, constant through time and across solved longitudes.
* Arrays of length `nlon` (the constructor's full longitude count), interpreted
  as Carrington profiles at bin centres, like `v_boundary`.
* A `time` array and values of shape `(nt,)`, shared across longitudes, or
  `(nt, model.nlon)` for the actual solved longitudes. Times are seconds from
  the simulation start, including negative spin-up times. Linear interpolation
  is used, with endpoint holding outside the provided interval. Time samples
  and simulation resolution must resolve the features being injected.

For example, this transverse magnetic pulse can accompany a `ConeCME` or a
prescribed `input_v_ts` plasma series:

```python
magnetic_pulse = {
    'br': 0*u.nT,
    'time': [-10, 0, .25, .5, .75, 5]*u.day,
    'bt': [-20, -20, -20, -100, -20, -20]*u.nT,
}
# Pass mhd_boundary=magnetic_pulse to SURF(...).
```

ConeCMEs continue to specify plasma enhancements and tracer trajectories.
They do not automatically inject a magnetic flux rope. Magnetic pulses must
be specified explicitly; a 1D time series does not describe a 3D flux rope.
SURF's synodic/sidereal setting controls boundary longitude mapping; it does
not add transverse motion or Coriolis forces.

The solver reuses SURF's spin-up, output schedule, radial tracers, independent
longitude slices, threaded execution, CME/streak/HCS tracking, and plasma
analysis functions. `get_observer_timeseries` and `get_SURF_at_position_HEEQ`
also return `br`, `bt`, `bn` in nT. HDF5 saves include
the physical vector grids and boundary configuration. `get_final_state`,
`set_initial_state` and `solve_chunked` retain all MHD components and preserve
the boundary time origin. Restarts require compatible radial flux and grids.
`SURF3d` can pass the same MHD configuration to each independent latitude slice;
it does not introduce angular MHD coupling. The SURFs_UP web form has no
physical magnetic boundary controls; configure MHD through the Python API.

For low-level SI problems:

```python
from surf.surf_solvers import create_solver

solver = create_solver(
    np.linspace(2.1e10, 1.5e11, 256), method='mhd-plm-rk2',
    br=1e-8, boundary={'bt': lambda t: -2e-8}, gravity=False,
)
result = solver.solve(
    np.linspace(0, 86400, 25),
    v_bc_func=lambda t: 4e5,
    rho_bc_func=lambda t: 5e-19,
    T_bc_func=lambda t: 5e5,
)
```

Low-level `mhd_init` accepts radial profiles for `bt/bn`, together with
`v_init/rho_init/T_init` for restart. Default initialization uses constant vr,
rho proportional to r^-2, T proportional to r^[-2(gamma-1)], and transverse
fields proportional to r^-1. It is a startup estimate, not an
exact steady MHD wind; allow spin-up before interpreting the solution.

## Scope and verification

Prescribed inflow and extrapolated outflow are intended for super-fast outward
heliospheric flow. Sub-fast boundaries require characteristic boundary conditions
that are not implemented here. Gravity is optional and defaults off to retain
SURF hydro's propagation assumptions; switching it on does not add coronal
heating, heat conduction, radiative losses, or a transcritical wind solution.
There is no reconnection/resistivity, multi-fluid/PUI thermodynamics, or lateral
stream deflection. The hydro empirical PUI option is explicitly rejected by
the MHD factory rather than silently applying an incompatible energy update.

`tests/test_mhd_solver.py` checks SI conversion and fluxes, magnetosonic limits,
fixed radial flux and radial force balance, shell conservation budgets, gravity
work, spherical density/transverse-field expansion, an analytic travelling
Alfven wave with resolution convergence, Brio-Wu shock evolution, and an
independently constructed perpendicular fast shock satisfying the
Rankine-Hugoniot jump conditions. Integration tests cover magnetic time series,
chunk/restart agreement, file round trips, parallel longitude slices and CME
tracers. These provide numerical verification; event-level validation against
observations and other MHD codes remains future scientific work.
