"""One-dimensional spherical hydrodynamic solvers for SURF."""

import numpy as np
from numba import njit

from surf.surf import surf_constants

__all__ = [
    'CompressibleSolver',
    'create_solver',
    'benchmark_solvers',
    'list_available_methods',
    'K_B_SI',
    'M_P_SI',
]

_CONSTANTS = surf_constants()
K_B_SI = _CONSTANTS['boltzmann_constant']
M_P_SI = _CONSTANTS['proton_mass']
SMALL_RHO = _CONSTANTS['min_density']
SMALL_P = _CONSTANTS['min_pressure']
NUMBA_CACHE = _CONSTANTS['numba_cache']
AU_M = 1.496e11
PUI_REFERENCE_RADIUS_M = AU_M
PUI_SLOWDOWN_PER_AU = 0.0027
TWO_PI = 2 * np.pi
del _CONSTANTS


# Particle advection

@njit(cache=NUMBA_CACHE)
def _advect_cme_particle_huxt(r_p, v_grid, r_grid, dt):
    """Advance a CME tracer using HUXt's integration convention.

    Parameters
    ----------
    r_p : float
        Current radius, in m.
    v_grid, r_grid : ndarray
        Radial velocity and position grids, in m/s and m.
    dt : float
        Time step, in s.

    Returns
    -------
    tuple of float
        Updated radius and sampled velocity.
    """
    dr = r_grid[1] - r_grid[0]
    v_p = np.interp(r_p - 0.5 * dr, r_grid, v_grid)
    r_new = r_p + v_p * dt
    if r_new > r_grid[-1]:
        r_new = r_grid[-1]
    return r_new, v_p


@njit(cache=NUMBA_CACHE)
def _advance_particles_huxt_batch(r_particles, v_particles, active, injected,
                                  injection_times, release_times, behaviors,
                                  require_positive_time, U, r_grid, time, dt):
    """Advance all hydro tracers in one compiled HUXt-style batch.

    Parameters
    ----------
    r_particles, v_particles : ndarray
        Particle radii and velocities, updated in place.
    active, injected : ndarray
        Particle state flags, updated in place.
    injection_times, release_times : ndarray
        Injection and release times, in s.
    behaviors, require_positive_time : ndarray
        Boundary behavior codes and positive-time requirements.
    U : ndarray
        Conserved hydrodynamic state.
    r_grid : ndarray
        Radial grid, in m.
    time, dt : float
        Current time and time step, in s.

    Returns
    -------
    None
        Input particle arrays are modified in place.
    """
    dr = r_grid[1] - r_grid[0]
    n_grid = r_grid.size
    for particle_id in range(r_particles.size):
        if not injected[particle_id]:
            time_is_valid = (not require_positive_time[particle_id]
                             or time > 0.0)
            if time >= injection_times[particle_id] and time_is_valid:
                r_particles[particle_id] = r_grid[0]
                active[particle_id] = True
                injected[particle_id] = True
            else:
                continue

        if not active[particle_id]:
            continue

        if time <= release_times[particle_id]:
            r_particles[particle_id] = r_grid[0]

        sample_r = r_particles[particle_id] - 0.5 * dr
        if sample_r <= r_grid[0]:
            velocity = U[0, 1] / U[0, 0]
        elif sample_r >= r_grid[-1]:
            velocity = U[-1, 1] / U[-1, 0]
        else:
            upper = np.searchsorted(r_grid, sample_r)
            lower = upper - 1
            weight = ((sample_r - r_grid[lower]) /
                      (r_grid[upper] - r_grid[lower]))
            v_lower = U[lower, 1] / U[lower, 0]
            v_upper = U[upper, 1] / U[upper, 0]
            velocity = v_lower + weight * (v_upper - v_lower)

        radius = r_particles[particle_id] + velocity * dt
        if radius > r_grid[n_grid - 1]:
            if behaviors[particle_id] == 1:
                radius = r_grid[n_grid - 1]
            else:
                active[particle_id] = False
        r_particles[particle_id] = radius
        v_particles[particle_id] = velocity


@njit(cache=NUMBA_CACHE)
def _catch_up_particles_huxt_batch(
        r_particles, v_particles, active, injected, injection_times,
        release_times, behaviors, require_positive_time, U, r_grid,
        particle_step_times, particle_step_idx, previous_particle_time,
        plasma_time):
    """Advance fixed-clock particle steps crossed by a hydro CFL step.

    Parameters
    ----------
    r_particles, v_particles, active, injected : ndarray
        Particle state arrays, updated in place.
    injection_times, release_times : ndarray
        Injection and release times, in s.
    behaviors, require_positive_time : ndarray
        Boundary behavior codes and positive-time requirements.
    U, r_grid : ndarray
        Conserved hydrodynamic state and radial grid.
    particle_step_times : ndarray
        Fixed particle-update times, in s.
    particle_step_idx : int
        Index of the next particle update.
    previous_particle_time, plasma_time : float
        Previous particle time and current plasma time, in s.

    Returns
    -------
    tuple
        Next particle-step index and most recent particle time.
    """
    tolerance = 1e-7 * max(1.0, abs(plasma_time))
    while (particle_step_idx < particle_step_times.size
           and particle_step_times[particle_step_idx] <= plasma_time + tolerance):
        particle_time = particle_step_times[particle_step_idx]
        particle_dt = particle_time - previous_particle_time
        _advance_particles_huxt_batch(
            r_particles, v_particles, active, injected, injection_times,
            release_times, behaviors, require_positive_time, U, r_grid,
            particle_time, particle_dt)
        previous_particle_time = particle_time
        particle_step_idx += 1
    return particle_step_idx, previous_particle_time


# Boundary conditions

@njit(cache=NUMBA_CACHE)
def _eval_boundary(mode, xs, ys, params, t):
    """Evaluate a tabulated boundary condition at a model time.

    Parameters
    ----------
    mode : int
        0 for a constant ``ys[0]``; 1 for a time series sampled at
        ``xs + params[0]``; 2 for a Carrington longitude map, with ``xs`` and
        ``ys`` padded as by :func:`periodic_boundary`.
    xs, ys : ndarray
        Sample positions and values.
    params : ndarray
        Time offset, longitude at the offset time (rad) and angular speed
        (rad/s). Only the time offset is used for mode 1.
    t : float
        Model time, in s.

    Returns
    -------
    float
        Boundary value at ``t``.
    """
    if mode == 0:
        return ys[0]
    if mode == 1:
        return np.interp(t + params[0], xs, ys)
    lon = (params[1] - params[2] * (t + params[0])) % TWO_PI
    return np.interp(lon, xs, ys)


def constant_boundary(value):
    """Return the tabulated form of a constant boundary condition.

    Parameters
    ----------
    value : float
        Boundary value.

    Returns
    -------
    tuple
        ``(mode, xs, ys, params)`` for :func:`_eval_boundary`.
    """
    return 0, np.zeros(1), np.array([float(value)]), np.zeros(3)


def time_series_boundary(times, values, offset=0.0):
    """Return the tabulated form of a boundary time series.

    Parameters
    ----------
    times, values : array_like
        Increasing sample times, in s, and boundary values.
    offset : float, optional
        Added to the model time before interpolating, in s.

    Returns
    -------
    tuple
        ``(mode, xs, ys, params)`` for :func:`_eval_boundary`.
    """
    return (1, np.ascontiguousarray(times, dtype=float),
            np.ascontiguousarray(values, dtype=float), np.array([float(offset), 0.0, 0.0]))


def periodic_boundary(lons, values, lon0, omega, offset=0.0):
    """Return the tabulated form of a Carrington longitude map.

    The boundary at time ``t`` is the map at longitude ``lon0 - omega * (t + offset)``,
    interpolated periodically, as by ``np.interp(..., period=2*pi)``.

    Parameters
    ----------
    lons, values : array_like
        Map longitudes, in rad, and boundary values.
    lon0 : float
        Longitude at time ``-offset``, in rad.
    omega : float
        Rotation rate, in rad/s.
    offset : float, optional
        Added to the model time, in s.

    Returns
    -------
    tuple
        ``(mode, xs, ys, params)`` for :func:`_eval_boundary`.
    """
    xs = np.asarray(lons, dtype=float) % TWO_PI
    order = np.argsort(xs)
    xs = xs[order]
    ys = np.asarray(values, dtype=float)[order]
    xs = np.concatenate((xs[-1:] - TWO_PI, xs, xs[:1] + TWO_PI))
    ys = np.concatenate((ys[-1:], ys, ys[:1]))
    return 2, xs, ys, np.array([float(offset), float(lon0), float(omega)])


def _plasma_boundary(t_grid, v_bc, rho_bc, T_bc, bc_times):
    """Check and tabulate the velocity, density and temperature boundary conditions.

    Parameters
    ----------
    t_grid : ndarray
        Output times, in s.
    v_bc, rho_bc, T_bc : float or array_like
        Boundary values, constant or sampled at ``bc_times``.
    bc_times : array_like or None
        Sample times of array boundary conditions, in s.

    Returns
    -------
    tuple of ndarray
        Sample times, velocity, density and temperature, as contiguous float arrays.
    """
    values = (v_bc, rho_bc, T_bc)
    if any(callable(value) for value in values):
        raise TypeError('Boundary conditions must be numbers, or arrays sampled at bc_times')
    if bc_times is None:
        if any(np.ndim(value) != 0 for value in values):
            raise ValueError('bc_times is required for array boundary conditions')
        bc_times = [t_grid[0]]
    bc_times = np.ascontiguousarray(bc_times, dtype=float)
    if bc_times.ndim != 1 or np.any(np.diff(bc_times) <= 0):
        raise ValueError('bc_times must be one-dimensional and strictly increasing')
    tabulated = []
    for value in values:
        value = np.ascontiguousarray(np.broadcast_to(value, bc_times.shape), dtype=float)
        tabulated.append(value)
    return (bc_times, *tabulated)


# Equation of state

@njit(cache=NUMBA_CACHE)
def _prim_to_cons(rho, v, p, gamma):
    """Convert primitive variables to conserved variables.

    Parameters
    ----------
    rho, v, p : float
        Density, velocity and pressure.
    gamma : float
        Adiabatic index.

    Returns
    -------
    tuple of float
        Conserved state ``(rho, rho*v, E)``.
    """
    mom = rho * v
    rhoe = p / (gamma - 1.0)
    ener = rhoe + 0.5 * rho * v**2
    return rho, mom, ener


# HLLC Riemann solver

@njit(cache=NUMBA_CACHE)
def _riemann_hllc(rho_l, mom_l, E_l, rho_r, mom_r, E_r, gamma):
    """Return the HLLC interface flux for two conserved states.

    Parameters
    ----------
    rho_l, mom_l, E_l : float
        Left conserved state.
    rho_r, mom_r, E_r : float
        Right conserved state.
    gamma : float
        Adiabatic index.

    Returns
    -------
    tuple of float
        Mass, momentum, and energy fluxes.
    """
    U_l0 = rho_l
    U_r0 = rho_r
    # Left state primitives
    rho_l = max(U_l0, SMALL_RHO)
    v_l = mom_l / rho_l
    p_l = max((E_l - 0.5*rho_l*v_l**2) * (gamma - 1.0), SMALL_P)
    c_l = np.sqrt(gamma * p_l / rho_l)

    # Right state primitives
    rho_r = max(U_r0, SMALL_RHO)
    v_r = mom_r / rho_r
    p_r = max((E_r - 0.5*rho_r*v_r**2) * (gamma - 1.0), SMALL_P)
    c_r = np.sqrt(gamma * p_r / rho_r)

    # Wave speed estimates
    S_l = min(v_l - c_l, v_r - c_r)
    S_r = max(v_l + c_l, v_r + c_r)

    if S_l >= 0:
        # Supersonic flow to the right
        return rho_l * v_l, rho_l * v_l * v_l + p_l, (E_l + p_l) * v_l
    elif S_r <= 0:
        # Supersonic flow to the left
        return rho_r * v_r, rho_r * v_r * v_r + p_r, (E_r + p_r) * v_r

    # Contact wave speed
    denom = rho_l * (S_l - v_l) - rho_r * (S_r - v_r)
    if abs(denom) < 1e-14:
        S_c = 0.5 * (v_l + v_r)
    else:
        S_c = (p_r - p_l + rho_l * v_l * (S_l - v_l) - rho_r * v_r * (S_r - v_r)) / denom

    if S_c >= 0:
        # Left star state
        factor = rho_l * (S_l - v_l) / (S_l - S_c)
        U_star_0 = factor
        U_star_1 = factor * S_c
        U_star_2 = factor * (E_l/rho_l + (S_c - v_l) * (S_c + p_l/(rho_l*(S_l - v_l))))
        return (rho_l * v_l + S_l * (U_star_0 - U_l0),
                rho_l * v_l * v_l + p_l + S_l * (U_star_1 - mom_l),
                (E_l + p_l) * v_l + S_l * (U_star_2 - E_l))
    # Right star state
    factor = rho_r * (S_r - v_r) / (S_r - S_c)
    U_star_0 = factor
    U_star_1 = factor * S_c
    U_star_2 = factor * (E_r/rho_r + (S_c - v_r) * (S_c + p_r/(rho_r*(S_r - v_r))))
    return (rho_r * v_r + S_r * (U_star_0 - U_r0),
            rho_r * v_r * v_r + p_r + S_r * (U_star_1 - mom_r),
            (E_r + p_r) * v_r + S_r * (U_star_2 - E_r))


@njit(cache=NUMBA_CACHE)
def _set_flux(fluxes, i, U_l, U_r, gamma):
    """Store the HLLC flux between two rows of conserved states.

    Parameters
    ----------
    fluxes : ndarray
        Interface fluxes, updated in place.
    i : int
        Interface index.
    U_l, U_r : ndarray
        Left and right conserved states.
    gamma : float
        Adiabatic index.

    Returns
    -------
    None
    """
    fluxes[i, 0], fluxes[i, 1], fluxes[i, 2] = _riemann_hllc(
        U_l[0], U_l[1], U_l[2], U_r[0], U_r[1], U_r[2], gamma)


# Reconstruction

@njit(cache=NUMBA_CACHE)
def _mc_limiter(a, b):
    """Return the monotonized-central slope for two differences.

    Parameters
    ----------
    a, b : float
        Adjacent one-sided differences.

    Returns
    -------
    float
        Limited central slope.
    """
    c = 0.5 * (a + b)
    if a * b <= 0:
        return 0.0
    return np.sign(c) * min(abs(c), 2 * abs(a), 2 * abs(b))


@njit(cache=NUMBA_CACHE)
def _compute_fluxes_pcm(U, nr, gamma, fluxes):
    """Compute first-order piecewise-constant interface fluxes.

    Parameters
    ----------
    U : ndarray
        Conserved cell states.
    nr : int
        Number of radial cells.
    gamma : float
        Adiabatic index.
    fluxes : ndarray
        HLLC flux at each cell interface, filled in place.

    Returns
    -------
    None
    """
    _set_flux(fluxes, 0, U[0], U[0], gamma)
    for i in range(1, nr):
        _set_flux(fluxes, i, U[i-1], U[i], gamma)
    _set_flux(fluxes, nr, U[nr-1], U[nr-1], gamma)


@njit(cache=NUMBA_CACHE)
def _plm_face_flux(fluxes, i, q_l, dq_l, q_r, dq_r, gamma):
    """Store the HLLC flux between two reconstructed primitive states.

    The face states are ``q_l + 0.5 * dq_l`` and ``q_r - 0.5 * dq_r``, floored to the
    minimum density and pressure.

    Parameters
    ----------
    fluxes : ndarray
        Interface fluxes, updated in place.
    i : int
        Interface index.
    q_l, dq_l, q_r, dq_r : tuple of float
        Cell primitives ``(rho, v, p)`` and their limited slopes either side of the face.
    gamma : float
        Adiabatic index.

    Returns
    -------
    None
    """
    rho_l = max(q_l[0] + 0.5 * dq_l[0], SMALL_RHO)
    v_l = q_l[1] + 0.5 * dq_l[1]
    p_l = max(q_l[2] + 0.5 * dq_l[2], SMALL_P)
    rho_r = max(q_r[0] - 0.5 * dq_r[0], SMALL_RHO)
    v_r = q_r[1] - 0.5 * dq_r[1]
    p_r = max(q_r[2] - 0.5 * dq_r[2], SMALL_P)
    U_l = _prim_to_cons(rho_l, v_l, p_l, gamma)
    U_r = _prim_to_cons(rho_r, v_r, p_r, gamma)
    fluxes[i, 0], fluxes[i, 1], fluxes[i, 2] = _riemann_hllc(
        U_l[0], U_l[1], U_l[2], U_r[0], U_r[1], U_r[2], gamma)


@njit(cache=NUMBA_CACHE)
def _compute_fluxes_plm(U, nr, gamma, fluxes, q_all, q_valid):
    """Compute second-order piecewise-linear interface fluxes.

    Parameters
    ----------
    U : ndarray
        Conserved cell states.
    nr : int
        Number of radial cells.
    gamma : float
        Adiabatic index.
    fluxes : ndarray
        Reconstructed HLLC flux at each cell interface, filled in place.
    q_all, q_valid : ndarray
        Work arrays for the cell primitives and their validity.

    Returns
    -------
    None
    """
    # Convert all cells to primitives [rho, v, p]
    for j in range(nr):
        rho = max(U[j, 0], SMALL_RHO)
        v = U[j, 1] / rho
        eint = (U[j, 2] - 0.5 * rho * v * v) / rho
        p = max(rho * eint * (gamma - 1.0), SMALL_P)
        q_all[j, 0] = rho
        q_all[j, 1] = v
        q_all[j, 2] = p
        q_valid[j] = 0 if (rho <= 0 or p <= 0) else 1

    # Inner boundary
    _set_flux(fluxes, 0, U[0], U[0], gamma)

    # First interior face: one-sided slope in the boundary cell
    if nr >= 2:
        if nr < 3 or q_valid[0] == 0 or q_valid[1] == 0 or q_valid[2] == 0:
            _set_flux(fluxes, 1, U[0], U[1], gamma)
        else:
            q0 = (q_all[0, 0], q_all[0, 1], q_all[0, 2])
            q1 = (q_all[1, 0], q_all[1, 1], q_all[1, 2])
            slope_0 = (q_all[1, 0] - q_all[0, 0], q_all[1, 1] - q_all[0, 1],
                       q_all[1, 2] - q_all[0, 2])
            dq_1 = (_mc_limiter(q_all[1, 0] - q_all[0, 0], q_all[2, 0] - q_all[1, 0]),
                    _mc_limiter(q_all[1, 1] - q_all[0, 1], q_all[2, 1] - q_all[1, 1]),
                    _mc_limiter(q_all[1, 2] - q_all[0, 2], q_all[2, 2] - q_all[1, 2]))
            _plm_face_flux(fluxes, 1, q0, slope_0, q1, dq_1, gamma)

    # Standard internal interfaces
    for i in range(2, nr):
        idx = i - 1  # left cell of interface
        if q_valid[idx-1] == 0 or q_valid[idx] == 0 or q_valid[idx+1] == 0:
            _set_flux(fluxes, i, U[idx], U[idx+1], gamma)
            continue
        dq_l = (_mc_limiter(q_all[idx, 0] - q_all[idx-1, 0], q_all[idx+1, 0] - q_all[idx, 0]),
                _mc_limiter(q_all[idx, 1] - q_all[idx-1, 1], q_all[idx+1, 1] - q_all[idx, 1]),
                _mc_limiter(q_all[idx, 2] - q_all[idx-1, 2], q_all[idx+1, 2] - q_all[idx, 2]))
        if idx < nr - 2:
            dq_r = (_mc_limiter(q_all[idx+1, 0] - q_all[idx, 0],
                                q_all[idx+2, 0] - q_all[idx+1, 0]),
                    _mc_limiter(q_all[idx+1, 1] - q_all[idx, 1],
                                q_all[idx+2, 1] - q_all[idx+1, 1]),
                    _mc_limiter(q_all[idx+1, 2] - q_all[idx, 2],
                                q_all[idx+2, 2] - q_all[idx+1, 2]))
        else:
            dq_r = (0.0, 0.0, 0.0)
        q_l = (q_all[idx, 0], q_all[idx, 1], q_all[idx, 2])
        q_r = (q_all[idx+1, 0], q_all[idx+1, 1], q_all[idx+1, 2])
        _plm_face_flux(fluxes, i, q_l, dq_l, q_r, dq_r, gamma)

    # Outer boundary - 2nd order extrapolation
    if nr >= 2:
        rho_f = max(q_all[nr-1, 0] + 0.5 * (q_all[nr-1, 0] - q_all[nr-2, 0]), SMALL_RHO)
        v_f = q_all[nr-1, 1] + 0.5 * (q_all[nr-1, 1] - q_all[nr-2, 1])
        p_f = max(q_all[nr-1, 2] + 0.5 * (q_all[nr-1, 2] - q_all[nr-2, 2]), SMALL_P)
        U_f = _prim_to_cons(rho_f, v_f, p_f, gamma)
        fluxes[nr, 0], fluxes[nr, 1], fluxes[nr, 2] = _riemann_hllc(
            U_f[0], U_f[1], U_f[2], U_f[0], U_f[1], U_f[2], gamma)
    else:
        _set_flux(fluxes, nr, U[nr-1], U[nr-1], gamma)


# Time integration

@njit(cache=NUMBA_CACHE)
def _get_dt(U, nr, min_dx, gamma, cfl):
    """Compute the CFL-limited time step.

    Parameters
    ----------
    U : ndarray
        Conserved cell states.
    nr : int
        Number of radial cells.
    min_dx : float
        Smallest cell width, in m.
    gamma, cfl : float
        Adiabatic index and CFL number.

    Returns
    -------
    float
        Stable time step, in s.
    """
    max_speed = 0.0
    for i in range(nr):
        rho = max(U[i, 0], SMALL_RHO)
        v = U[i, 1] / rho
        eint = (U[i, 2] - 0.5 * rho * v * v) / rho
        p = max(rho * eint * (gamma - 1.0), SMALL_P)
        c = np.sqrt(gamma * p / rho)
        speed = abs(v) + c
        if speed > max_speed:
            max_speed = speed

    if max_speed == 0:
        return 1.0

    return cfl * min_dx / max_speed


@njit(cache=NUMBA_CACHE)
def _apply_pui_source(U, r, nr, dt, enabled):
    """Apply the 1 AU-anchored PUI trend while preserving internal energy.

    Parameters
    ----------
    U : ndarray
        Conserved states, updated in place.
    r : ndarray
        Radial cell centres, in m.
    nr : int
        Number of radial cells.
    dt : float
        Time step, in s.
    enabled : bool
        Whether to apply the source term.

    Returns
    -------
    None
    """
    if not enabled:
        return
    for i in range(nr):
        if r[i] <= PUI_REFERENCE_RADIUS_M:
            continue
        rho = max(U[i, 0], SMALL_RHO)
        velocity = U[i, 1] / rho
        internal_energy = U[i, 2] - 0.5 * rho * velocity * velocity
        distance_au = max((r[i] - PUI_REFERENCE_RADIUS_M) / AU_M, 0.0)
        trend = max(1.0 - PUI_SLOWDOWN_PER_AU * distance_au, 1e-6)
        # The radius-dependent rate integrates to the paper's linear radial trend.
        rate_per_m = (PUI_SLOWDOWN_PER_AU / AU_M) / trend
        factor = max(1.0 - rate_per_m * abs(velocity) * dt, 0.0)
        velocity_new = velocity * factor
        U[i, 1] = rho * velocity_new
        U[i, 2] = internal_energy + 0.5 * rho * velocity_new * velocity_new


@njit(cache=NUMBA_CACHE)
def _step_euler(U, U_new, U_bc, nr, A, V, r, dt, gamma, use_plm, pui, fluxes, q_all, q_valid):
    """Take an Euler step including spherical geometry and PUI forcing.

    Parameters
    ----------
    U : ndarray
        Conserved cell states.
    U_new : ndarray
        Conserved state after the step, filled in place.
    U_bc : tuple of float
        Conserved inner-boundary state.
    nr : int
        Number of radial cells.
    A, V, r : ndarray
        Interface areas, cell volumes, and cell-centre radii.
    dt, gamma : float
        Time step and adiabatic index.
    use_plm, pui : bool
        Whether to use PLM reconstruction and PUI forcing.
    fluxes, q_all, q_valid : ndarray
        Work arrays.

    Returns
    -------
    None
    """
    if use_plm:
        _compute_fluxes_plm(U, nr, gamma, fluxes, q_all, q_valid)
    else:
        _compute_fluxes_pcm(U, nr, gamma, fluxes)

    for i in range(nr):
        if V[i] <= 0:
            U_new[i, :] = U[i, :]
            continue

        # Geometric source term (momentum equation only)
        rho = max(U[i, 0], SMALL_RHO)
        v = U[i, 1] / rho
        eint = (U[i, 2] - 0.5 * rho * v * v) / rho
        p = max(rho * eint * (gamma - 1.0), SMALL_P)
        S_mom = p * (A[i+1] - A[i]) / V[i]

        inv_V = 1.0 / V[i]
        for j in range(3):
            dUdt = -(A[i+1] * fluxes[i+1, j] - A[i] * fluxes[i, j]) * inv_V
            if j == 1:
                dUdt += S_mom
            U_new[i, j] = U[i, j] + dt * dUdt

        # Reset if unphysical
        if U_new[i, 0] < SMALL_RHO or U_new[i, 2] < SMALL_P:
            U_new[i, :] = U[i, :]

    U_new[0, 0], U_new[0, 1], U_new[0, 2] = U_bc
    _apply_pui_source(U_new, r, nr, dt, pui)


@njit(cache=NUMBA_CACHE, nogil=True)
def _run_hydro(U, time, t_grid, bc_times, bc_v, bc_rho, bc_T, r, A, V, min_dx, gamma, cfl,
               use_plm, rk2, pui, p_r, p_v, p_active, p_injected, p_inj_times, p_rel_times,
               p_behaviors, p_positive_time, p_step_times, v_out, rho_out, T_out,
               rec_r, rec_v, rec_t, rec_on):
    """Advance the hydrodynamic solution through every output time.

    Parameters
    ----------
    U : ndarray
        Conserved cell states, updated in place.
    time : float
        Initial time, in s.
    t_grid : ndarray
        Output times, in s.
    bc_times, bc_v, bc_rho, bc_T : ndarray
        Inner-boundary sample times and values in SI units.
    r, A, V : ndarray
        Cell-centre radii, interface areas and cell volumes.
    min_dx, gamma, cfl : float
        Smallest cell width, adiabatic index and CFL number.
    use_plm, rk2, pui : bool
        PLM reconstruction, RK2 integration and PUI forcing switches.
    p_r, p_v, p_active, p_injected : ndarray
        Tracer state, updated in place.
    p_inj_times, p_rel_times, p_behaviors, p_positive_time : ndarray
        Tracer injection and release times and boundary behaviour.
    p_step_times : ndarray
        Fixed tracer-update times, in s.
    v_out, rho_out, T_out : ndarray
        Plasma output at each output time, filled in place.
    rec_r, rec_v, rec_t, rec_on : ndarray
        Tracer position, speed and time at each output time, and whether the
        tracer was active, filled in place.

    Returns
    -------
    float
        Final time, in s.
    """
    nr = r.size
    nt = t_grid.size
    U1 = np.empty_like(U)
    U2 = np.empty_like(U)
    fluxes = np.empty((nr + 1, 3))
    q_all = np.empty((nr, 3))
    q_valid = np.empty(nr, dtype=np.int32)

    p_step_idx = np.searchsorted(p_step_times, time, side='right')
    previous_particle_time = time

    t_idx = 0
    if abs(t_grid[0] - time) < 1e-5:
        _hydro_snapshot(U, nr, gamma, time, 0, v_out, rho_out, T_out,
                        p_r, p_v, p_active, rec_r, rec_v, rec_t, rec_on)
        t_idx = 1

    while t_idx < nt:
        target_time = t_grid[t_idx]

        while time < target_time:
            dt = _get_dt(U, nr, min_dx, gamma, cfl)
            if time + dt > target_time:
                dt = target_time - time

            t_bc = time + dt
            rho_bc = np.interp(t_bc, bc_times, bc_rho)
            v_bc = np.interp(t_bc, bc_times, bc_v)
            T_bc = np.interp(t_bc, bc_times, bc_T)
            p_bc = rho_bc * K_B_SI * T_bc / M_P_SI
            U_bc = _prim_to_cons(rho_bc, v_bc, p_bc, gamma)

            if rk2:
                # Heun's method: average the state with the result of two Euler steps
                _step_euler(U, U1, U_bc, nr, A, V, r, dt, gamma, use_plm, pui,
                            fluxes, q_all, q_valid)
                _step_euler(U1, U2, U_bc, nr, A, V, r, dt, gamma, use_plm, pui,
                            fluxes, q_all, q_valid)
                for i in range(nr):
                    for j in range(3):
                        U[i, j] = 0.5 * (U[i, j] + U2[i, j])
                U[0, 0], U[0, 1], U[0, 2] = U_bc
            else:
                _step_euler(U, U1, U_bc, nr, A, V, r, dt, gamma, use_plm, pui,
                            fluxes, q_all, q_valid)
                U[:, :] = U1
            time += dt

            if (p_r.size
                    and p_step_idx < p_step_times.size
                    and p_step_times[p_step_idx] <= time + 1e-7 * max(1.0, abs(time))):
                p_step_idx, previous_particle_time = _catch_up_particles_huxt_batch(
                    p_r, p_v, p_active, p_injected, p_inj_times, p_rel_times,
                    p_behaviors, p_positive_time, U, r, p_step_times,
                    p_step_idx, previous_particle_time, time)

        _hydro_snapshot(U, nr, gamma, time, t_idx, v_out, rho_out, T_out,
                        p_r, p_v, p_active, rec_r, rec_v, rec_t, rec_on)
        t_idx += 1
    return time


@njit(cache=NUMBA_CACHE)
def _hydro_snapshot(U, nr, gamma, time, t_idx, v_out, rho_out, T_out,
                    p_r, p_v, p_active, rec_r, rec_v, rec_t, rec_on):
    """Store the plasma fields and active tracers at an output time.

    Parameters
    ----------
    U : ndarray
        Conserved cell states.
    nr : int
        Number of radial cells.
    gamma, time : float
        Adiabatic index and current time, in s.
    t_idx : int
        Output time index.
    v_out, rho_out, T_out : ndarray
        Plasma output, filled in place.
    p_r, p_v, p_active : ndarray
        Tracer positions, speeds and activity.
    rec_r, rec_v, rec_t, rec_on : ndarray
        Tracer output, filled in place.

    Returns
    -------
    None
    """
    for i in range(nr):
        rho = max(U[i, 0], SMALL_RHO)
        v = U[i, 1] / rho
        eint = (U[i, 2] - 0.5 * rho * v * v) / rho
        p = max(rho * eint * (gamma - 1.0), SMALL_P)
        v_out[t_idx, i] = v
        rho_out[t_idx, i] = rho
        T_out[t_idx, i] = p * M_P_SI / (rho * K_B_SI)
    _record_particles(time, t_idx, p_r, p_v, p_active, rec_r, rec_v, rec_t, rec_on)


@njit(cache=NUMBA_CACHE)
def _record_particles(time, t_idx, p_r, p_v, p_active, rec_r, rec_v, rec_t, rec_on):
    """Record the active tracers at an output time.

    Parameters
    ----------
    time : float
        Current time, in s.
    t_idx : int
        Output time index.
    p_r, p_v, p_active : ndarray
        Tracer positions, speeds and activity.
    rec_r, rec_v, rec_t, rec_on : ndarray
        Tracer output, filled in place.

    Returns
    -------
    None
    """
    for k in range(p_r.size):
        if p_active[k]:
            rec_on[k, t_idx] = True
            rec_r[k, t_idx] = p_r[k]
            rec_v[k, t_idx] = p_v[k]
            rec_t[k, t_idx] = time


class CompressibleSolver:
    """Solve the 1D spherical Euler equations with an HLLC flux.

    Parameters
    ----------
    r_grid : ndarray
        Radial cell centres, in m.
    gamma : float, optional
        Adiabatic index.
    cfl : float, optional
        CFL number. Defaults to 0.4 for PLM and 0.8 for PCM.
    reconstruction : str, optional
        ``'pcm'`` or ``'plm'``.
    time_integration : str, optional
        ``'euler'`` or ``'rk2'``.
    verbose : bool, optional
        Retained for compatibility; the solver does not write terminal output.
    """

    def __init__(self, r_grid, gamma=1.5, cfl=None,
                 reconstruction='plm', time_integration='euler',
                 verbose=False, pui=False):
        """Initialise a hydrodynamic solver.

        Parameters
        ----------
        r_grid : ndarray
            Radial cell centres, in m.
        gamma : float
            Adiabatic index.
        cfl : float, optional
            CFL number; selected from the reconstruction when omitted.
        reconstruction, time_integration : str
            Spatial reconstruction and time-integration methods.
        verbose : bool
            Compatibility flag; no terminal output is produced.
        pui : bool
            Whether to apply empirical pick-up-ion deceleration.

        Returns
        -------
        None
        """
        self.r = np.ascontiguousarray(r_grid, dtype=float)
        self.nr = len(r_grid)
        self.gamma = gamma
        self.verbose = verbose
        self.pui = bool(pui)

        self.reconstruction = reconstruction.lower()
        self.time_integration = time_integration.lower()

        # Set CFL based on reconstruction order
        if cfl is None:
            self.cfl = 0.4 if self.reconstruction == 'plm' else 0.8
        else:
            self.cfl = cfl

        # Grid geometry
        self.dr = np.diff(r_grid)
        self.r_int = np.zeros(self.nr + 1)
        self.r_int[1:-1] = 0.5 * (self.r[1:] + self.r[:-1])
        self.r_int[0] = self.r[0] - 0.5 * self.dr[0]
        self.r_int[-1] = self.r[-1] + 0.5 * self.dr[-1]
        self.dx = np.diff(self.r_int)

        # Interface areas and cell volumes (spherical geometry)
        self.A = 4.0 * np.pi * self.r_int**2
        self.V = 4.0/3.0 * np.pi * (self.r_int[1:]**3 - self.r_int[:-1]**3)

        # State array [rho, rho*v, E]
        self.U = np.zeros((self.nr, 3))
        self.time = 0.0

        # Precompute min_dx for CFL
        self.min_dx = np.min(self.dx)

    def _set_initial_conditions(self, rho, v, T):
        """Set initial conditions from primitive variables.

        Parameters
        ----------
        rho, v, T : ndarray
            Density, velocity, and temperature profiles in SI units.

        Returns
        -------
        None
            ``self.U`` is updated in place.
        """
        for i in range(self.nr):
            p = rho[i] * K_B_SI * T[i] / M_P_SI
            p = max(p, SMALL_P)
            self.U[i] = _prim_to_cons(rho[i], v[i], p, self.gamma)

    def _run(self, t_grid, boundary, particles, outputs):
        """Advance the solution through the output times in compiled code.

        Parameters
        ----------
        t_grid : ndarray
            Output times, in s.
        boundary : tuple of ndarray
            Boundary sample times, velocity, density and temperature.
        particles : tuple of ndarray
            Packed tracer state and settings.
        outputs : tuple of ndarray
            Plasma and tracer output arrays, filled in place.

        Returns
        -------
        None
            ``self.U`` and ``self.time`` are updated.
        """
        self.time = _run_hydro(
            self.U, self.time, t_grid, *boundary, self.r, self.A, self.V, self.min_dx,
            self.gamma, self.cfl, self.reconstruction == 'plm',
            self.time_integration == 'rk2', self.pui, *particles, *outputs)

    def solve(self, t_grid, v_bc, rho_bc, T_bc, bc_times=None,
              num_particles=None, particle_injection_rate=None, particle_release_rate=None,
              particle_initial_positions=None,
              v_init=None, rho_init=None, T_init=None,
              particle_step_times=None):
        """Run the solver and optional tracers over an output-time grid.

        Parameters
        ----------
        t_grid : ndarray
            Output times, in s.
        v_bc : float or ndarray
            Velocity boundary condition, in m/s. Constant, or sampled at ``bc_times``.
        rho_bc : float or ndarray
            Density boundary condition, in kg/m³. Constant, or sampled at ``bc_times``.
        T_bc : float or ndarray
            Temperature boundary condition, in K. Constant, or sampled at ``bc_times``.
        bc_times : ndarray, optional
            Increasing sample times of array boundary conditions, in s. The boundary is
            interpolated linearly between them and held constant outside them.
        num_particles : dict or None, optional
            Number of tracers in each named group. None for no tracers.
        particle_injection_rate : dict, optional
            Injection times, in s, by group.
        particle_release_rate : dict, optional
            Release times, in s, by group. Defaults to the injection times.
        particle_initial_positions : dict, optional
            Initial tracer radii by group, in m.
        v_init, rho_init, T_init : ndarray, optional
            Initial plasma profiles. When all three are supplied they restart
            the solution; otherwise profiles are derived from boundary values.
        particle_step_times : ndarray, optional
            Fixed tracer-update times, in s. Defaults to ``t_grid``.

        Returns
        -------
        dict
            Plasma results and, when requested, particle trajectories.
        """
        t_grid = np.ascontiguousarray(t_grid, dtype=float)
        boundary = _plasma_boundary(t_grid, v_bc, rho_bc, T_bc, bc_times)
        bc_times, bc_v, bc_rho, bc_T = boundary

        # Initialize state: use provided profiles or power-law scaling
        if v_init is not None and rho_init is not None and T_init is not None:
            self._set_initial_conditions(rho_init, v_init, T_init)
        else:
            v0 = np.interp(t_grid[0], bc_times, bc_v)
            rho0 = np.interp(t_grid[0], bc_times, bc_rho)
            T0 = np.interp(t_grid[0], bc_times, bc_T)

            r0 = self.r[0]
            v_def = np.ones_like(self.r) * v0
            rho_def = np.zeros_like(self.r)
            T_def = np.zeros_like(self.r)
            alpha = 2 * (self.gamma - 1)

            for i, r in enumerate(self.r):
                r_ratio = r0 / r
                rho_def[i] = rho0 * r_ratio**2
                T_def[i] = T0 * r_ratio**alpha

            self._set_initial_conditions(rho_def, v_def, T_def)
        self.time = t_grid[0]

        # Pack every named tracer group into flat arrays, so the compiled loop
        # updates CME, HCS and streakline tracers together.
        num_particles = dict(num_particles or {})
        v_initial = self.U[:, 1] / self.U[:, 0]
        p_r, p_v, p_active, p_inj_times, p_rel_times = [], [], [], [], []
        p_behaviors, p_positive_time = [], []
        groups = {}
        for group_name, n_p in num_particles.items():
            inj_times = list(particle_injection_rate[group_name])
            rel_times = (particle_release_rate[group_name]
                         if particle_release_rate and group_name in particle_release_rate
                         else inj_times)
            is_cme = 'cme' in group_name.lower()
            initial_positions = list(
                particle_initial_positions.get(group_name, [])
                if particle_initial_positions else [])
            groups[group_name] = {
                'n_particles': n_p,
                'injection_times': inj_times,
                'release_times': rel_times,
                'start': len(p_r),
            }
            for particle_id in range(n_p):
                has_initial = particle_id < len(initial_positions)
                r_p = initial_positions[particle_id] if has_initial else np.nan
                p_r.append(r_p)
                p_v.append(np.interp(r_p, self.r, v_initial) if has_initial else np.nan)
                p_active.append(has_initial)
                p_inj_times.append(inj_times[particle_id])
                p_rel_times.append(rel_times[particle_id])
                p_behaviors.append(1 if is_cme else 0)
                p_positive_time.append(is_cme)

        p_active = np.asarray(p_active, dtype=bool)
        if particle_step_times is None:
            particle_step_times = t_grid
        particles = (
            np.asarray(p_r, dtype=float), np.asarray(p_v, dtype=float),
            p_active, p_active.copy(),
            np.asarray(p_inj_times, dtype=float), np.asarray(p_rel_times, dtype=float),
            np.asarray(p_behaviors, dtype=np.int64), np.asarray(p_positive_time, dtype=bool),
            np.ascontiguousarray(particle_step_times, dtype=float))

        nt = len(t_grid)
        n_tracers = len(p_r)
        outputs = (np.zeros((nt, self.nr)), np.zeros((nt, self.nr)), np.zeros((nt, self.nr)),
                   np.full((n_tracers, nt), np.nan), np.full((n_tracers, nt), np.nan),
                   np.full((n_tracers, nt), np.nan), np.zeros((n_tracers, nt), dtype=bool))
        self._run(t_grid, boundary, particles, outputs)
        v_out, rho_out, T_out, rec_r, rec_v, rec_t, rec_on = outputs

        results = {
            't': t_grid,
            'r': self.r,
            'v': v_out,
            'rho': rho_out,
            'T': T_out,
            'solver_info': {
                'reconstruction': self.reconstruction,
                'time_integration': self.time_integration,
                'pui': self.pui,
            }
        }

        # Each tracer's trajectory holds the output times at which it was active,
        # padded with NaN to the longest trajectory in its group.
        if groups:
            p_active, p_injected = particles[2], particles[3]
            for group in groups.values():
                start = group.pop('start')
                stop = start + group['n_particles']
                lengths = [int(np.count_nonzero(rec_on[k])) for k in range(start, stop)]
                max_len = max(lengths, default=0)
                for key, record in (('r', rec_r), ('v', rec_v), ('t', rec_t)):
                    trajectories = np.full((len(lengths), max_len), np.nan)
                    for row, k in enumerate(range(start, stop)):
                        trajectories[row, :lengths[row]] = record[k, rec_on[k]]
                    group[key] = trajectories
                group['t_inject'] = np.array(group['injection_times'])
                group['active'] = p_active[start:stop]
                group['particles_injected'] = int(np.sum(p_injected[start:stop]))
            results['particles'] = {'groups': groups}

        return results


def create_solver(r_grid, gamma=1.5, method='hllc-plm', cfl=None, verbose=False,
                  pui=False, **kwargs):
    """Create a configured hydrodynamic or MHD solver.
    
    Parameters
    ----------
    r_grid : ndarray
        Radial grid, in m.
    gamma : float, optional
        Adiabatic index.
    method : str, optional
        ``'hllc-{pcm|plm}[-{euler|rk2}]'`` for hydrodynamics or
        ``'mhd-{pcm|plm}[-{euler|rk2}]'`` for ideal MHD.
    cfl : float, optional
        CFL number. The default depends on the numerical method.
    verbose : bool, optional
        Retained for compatibility; solvers do not write terminal output.
    pui : bool, optional
        Apply empirical pick-up-ion deceleration to hydrodynamic solutions.
    **kwargs
        Additional MHD solver options.
        
    Returns
    -------
    CompressibleSolver or MHDSolver
        Configured solver.
    """
    parts = method.lower().split('-')

    if parts[0] == 'mhd':
        from surf.surf_mhd import MHDSolver
        if len(parts) > 3:
            raise ValueError('MHD method format is mhd-{pcm|plm}-{euler|rk2}')
        return MHDSolver(
            r_grid, gamma=gamma, cfl=cfl, verbose=verbose, pui=pui,
            reconstruction=parts[1] if len(parts) > 1 else 'plm',
            time_integration=parts[2] if len(parts) > 2 else 'rk2', **kwargs)
    if kwargs:
        raise TypeError(f"Unexpected hydro solver options: {', '.join(kwargs)}")
    
    reconstruction = parts[1] if len(parts) > 1 else 'plm'
    time_int = parts[2] if len(parts) > 2 else 'euler'
    
    return CompressibleSolver(
        r_grid, gamma=gamma, cfl=cfl,
        reconstruction=reconstruction,
        time_integration=time_int, verbose=verbose, pui=pui
    )


def benchmark_solvers(r_grid, t_end, v_bc, rho_bc, T_bc, gamma=1.5,
                      methods=None, reference='hllc-plm-rk2'):
    """Compare solver configurations on a constant-boundary problem.
    
    Parameters
    ----------
    r_grid : ndarray
        Radial grid, in m.
    t_end : float
        End time, in s.
    v_bc, rho_bc, T_bc : float
        Constant boundary values.
    gamma : float, optional
        Adiabatic index.
    methods : list, optional
        Method strings to compare.
    reference : str, optional
        Method used as the error reference.
        
    Returns
    -------
    dict
        Results, timings, and errors by method.
    """
    import time as time_module
    
    if methods is None:
        methods = [
            'hllc-pcm',
            'hllc-plm',
            'hllc-plm-rk2',
        ]
    
    dt_out = t_end / 100
    t_grid = np.arange(0, t_end + dt_out, dt_out)
    
    results = {}
    
    print(f"Running reference: {reference}")
    solver = create_solver(r_grid, gamma=gamma, method=reference, verbose=False)
    t0 = time_module.time()
    ref_result = solver.solve(t_grid, v_bc, rho_bc, T_bc)
    ref_time = time_module.time() - t0
    results[reference] = {
        'result': ref_result,
        'time': ref_time,
        'error_v': 0.0,
        'error_rho': 0.0,
    }
    
    for method in methods:
        if method == reference:
            continue
            
        print(f"Running: {method}")
        solver = create_solver(r_grid, gamma=gamma, method=method, verbose=False)
        t0 = time_module.time()
        result = solver.solve(t_grid, v_bc, rho_bc, T_bc)
        elapsed = time_module.time() - t0
        
        error_v = np.sqrt(np.mean((result['v'][-1] - ref_result['v'][-1])**2))
        error_rho = np.sqrt(np.mean((result['rho'][-1] - ref_result['rho'][-1])**2))
        
        results[method] = {
            'result': result,
            'time': elapsed,
            'error_v': error_v,
            'error_rho': error_rho,
        }
    
    print("\n" + "="*70)
    print(f"{'Method':<20} {'Time (s)':<12} {'Error v':<15} {'Error rho':<15}")
    print("="*70)
    for method, data in results.items():
        print(f"{method:<20} {data['time']:<12.3f} {data['error_v']:<15.2e} "
              f"{data['error_rho']:<15.2e}")
    
    return results


def list_available_methods():
    """Print solver choices and return the low-level configuration options.

    SURF(solver='hydro') uses HLLC + PLM + RK2, whereas create_solver()
    defaults to HLLC + PLM + Euler. Method strings below apply to the
    low-level factory; they are not public SURF solver names.

    Returns
    -------
    dict
        Available physics, reconstruction, and time-integration options.
    """
    print("""Available compressible solvers
    
SURF names:
  hydro      HLLC + PLM + RK2       mhd      HLL + PLM + RK2
  hydro-pcm  HLLC + PCM + Euler     mhd-pcm  HLL + PCM + Euler
  hydro variants also accept the -pui suffix.
    
create_solver() methods:
  hllc[-{pcm|plm}[-{euler|rk2}]]
  mhd[-{pcm|plm}[-{euler|rk2}]]
  Defaults: hllc-plm uses Euler; mhd uses PLM + RK2.
    
SURF() defaults to the incompressible huxt solver.""")
    
    return {
        'physics': ['hydro', 'mhd'],
        'reconstruction': ['pcm', 'plm'],
        'time_integrator': ['euler', 'rk2'],
    }
