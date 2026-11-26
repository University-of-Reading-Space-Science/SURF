"""One-dimensional radial ideal-MHD solver for SURF.

Radial magnetic flux is fixed by ``r**2 Br = constant``. The conserved state is
``[rho, rho*vr, E, Bt, Bn]`` in SI units.
"""

import numpy as np
from numba import njit

from surf.surf_solvers import (
    CompressibleSolver,
    K_B_SI,
    M_P_SI,
    NUMBA_CACHE,
    SMALL_P,
    SMALL_RHO,
    _catch_up_particles_huxt_batch,
    _eval_boundary,
    _mc_limiter,
    _record_particles,
    constant_boundary,
    periodic_boundary,
    time_series_boundary,
)

MU0 = 4e-7 * np.pi
GM_SUN = 1.32712440018e20  # m^3 s^-2
MHD_COMPONENTS = ('br', 'bt', 'bn')


def normalize_boundary(boundary, nlon_full, nlon):
    """Validate an MHD boundary mapping and convert quantities to SI.

    Parameters
    ----------
    boundary : mapping or None
        Magnetic boundary values, optionally including sample times.
    nlon_full, nlon : int
        Full boundary size and number of solved longitudes.

    Returns
    -------
    dict
        Copied boundary arrays in tesla and seconds.
    """
    from astropy import units as u

    boundary = dict(boundary or {})
    unknown = set(boundary) - set(MHD_COMPONENTS) - {'time'}
    if unknown:
        raise ValueError(f'Unknown mhd_boundary keys: {sorted(unknown)}')

    result = {}
    for key, value in boundary.items():
        unit = u.s if key == 'time' else u.T
        value = np.asarray(
            value.to_value(unit) if isinstance(value, u.Quantity) else value,
            dtype=float,
        )
        if not np.all(np.isfinite(value)):
            raise ValueError(f'mhd_boundary[{key!r}] must be finite')
        result[key] = value.copy()

    times = result.get('time')
    if times is not None and (
            times.ndim != 1 or len(times) < 2
            or np.any(np.diff(times) <= 0)):
        raise ValueError('mhd_boundary time must contain at least two increasing seconds')

    for key in MHD_COMPONENTS:
        value = result.setdefault(key, np.asarray(0.0))
        if key == 'br' and value.ndim != 0:
            raise ValueError('br must be scalar and time independent in strict radial MHD')
        if value.ndim == 0:
            continue
        shapes = (
            [(nlon_full,)]
            if times is None
            else [(len(times),), (len(times), nlon)]
        )
        if value.shape not in shapes:
            raise ValueError(f'mhd_boundary[{key!r}] needs a scalar or shape in {shapes}')
    return result


def boundary_for_longitude(model, i):
    """Return SI boundary values for one SURF longitude.

    Parameters
    ----------
    model : SURF
        Model containing the normalized MHD boundary.
    i : int
        Longitude index.

    Returns
    -------
    dict
        Radial field, tabulated transverse boundaries, and gravity flag.
    """
    config = model.mhd_boundary
    offset = getattr(model, '_mhd_time_offset', 0.0)
    boundary = {}
    for key in ('bt', 'bn'):
        value = config[key]
        if value.ndim == 0:
            boundary[key] = float(value)
        elif 'time' in config:
            series = value if value.ndim == 1 else value[:, i]
            boundary[key] = time_series_boundary(config['time'], series, offset)
        else:
            # Carrington bin centres, same convention as v_boundary. Vectors
            # are inertial RTN components; frame only changes longitude mapping.
            lons = (np.arange(len(value)) + 0.5) * 2 * np.pi / len(value)
            lon0 = model.cr_lon_init.to_value('rad') + model.lon[i].to_value('rad')
            omega = 2 * np.pi / model.rotation_period.to_value('s')
            boundary[key] = periodic_boundary(lons, value, lon0, omega, offset)
    return {
        'br': float(config['br']),
        'boundary': boundary,
        'gravity': model.mhd_gravity,
    }


# Pointwise MHD kernels. States are tuples (rho, rho*vr, E, Bt, Bn) for conserved
# variables and (rho, vr, p, Bt, Bn) for primitives.

@njit(cache=NUMBA_CACHE)
def _primitive_s(U0, U1, U2, U3, U4, gamma):
    """Convert a conserved state to primitive variables."""
    kinetic = 0.5 * U1**2 / U0
    magnetic = (U3**2 + U4**2) / (2 * MU0)
    return U0, U1 / U0, (gamma - 1) * (U2 - kinetic - magnetic), U3, U4


@njit(cache=NUMBA_CACHE)
def _conserved_s(q0, q1, q2, q3, q4, gamma):
    """Convert a primitive state to conserved variables."""
    energy = (q2 / (gamma - 1)
              + 0.5 * q0 * q1**2
              + (q3**2 + q4**2) / (2 * MU0))
    return q0, q1 * q0, energy, q3, q4


@njit(cache=NUMBA_CACHE)
def _fast_speed_s(q0, q2, q3, q4, gamma):
    """Return the reduced system's radial fast-mode speed."""
    a2 = gamma * q2 / q0
    bt2 = (q3**2 + q4**2) / (MU0 * q0)
    # With transverse momentum removed, force-free Br has no radial wave
    # characteristic. Only gas and transverse magnetic pressure contribute.
    return np.sqrt(a2 + bt2)


@njit(cache=NUMBA_CACHE)
def _flux_s(q, gamma):
    """Return the physical flux for a primitive state tuple."""
    U = _conserved_s(q[0], q[1], q[2], q[3], q[4], gamma)
    pt = q[2] + (q[3]**2 + q[4]**2) / (2 * MU0)
    # The analytic radial background stress cancels its geometric source.
    return U[1], U[1] * q[1] + pt, (U[2] + pt) * q[1], q[1] * q[3], q[1] * q[4]


@njit(cache=NUMBA_CACHE)
def _hll_flux_s(ql, qr, gamma):
    """Return the two-wave HLL flux between two primitive state tuples."""
    cl = _fast_speed_s(ql[0], ql[2], ql[3], ql[4], gamma)
    cr = _fast_speed_s(qr[0], qr[2], qr[3], qr[4], gamma)
    sl = min(ql[1] - cl, qr[1] - cr)
    sr = max(ql[1] + cl, qr[1] + cr)
    fl = _flux_s(ql, gamma)
    fr = _flux_s(qr, gamma)
    if sl >= 0:
        return fl
    if sr <= 0:
        return fr
    Ul = _conserved_s(ql[0], ql[1], ql[2], ql[3], ql[4], gamma)
    Ur = _conserved_s(qr[0], qr[1], qr[2], qr[3], qr[4], gamma)
    width = sr - sl
    return ((sr * fl[0] - sl * fr[0] + sl * sr * (Ur[0] - Ul[0])) / width,
            (sr * fl[1] - sl * fr[1] + sl * sr * (Ur[1] - Ul[1])) / width,
            (sr * fl[2] - sl * fr[2] + sl * sr * (Ur[2] - Ul[2])) / width,
            (sr * fl[3] - sl * fr[3] + sl * sr * (Ur[3] - Ul[3])) / width,
            (sr * fl[4] - sl * fr[4] + sl * sr * (Ur[4] - Ul[4])) / width)


# Array forms of the kernels, for analysis and tests

@njit(cache=NUMBA_CACHE)
def _primitive(U, gamma):
    """Convert a conserved state to primitive variables.

    Parameters
    ----------
    U : ndarray
        Conserved state ``[rho, rho*vr, E, Bt, Bn]``.
    gamma : float
        Adiabatic index.

    Returns
    -------
    ndarray
        Primitive state ``[rho, vr, p, Bt, Bn]``.
    """
    q = np.empty(5)
    q[0], q[1], q[2], q[3], q[4] = _primitive_s(U[0], U[1], U[2], U[3], U[4], gamma)
    return q


@njit(cache=NUMBA_CACHE)
def _conserved(q, gamma):
    """Convert a primitive MHD state to conserved variables.

    Parameters
    ----------
    q : ndarray
        Primitive state ``[rho, vr, p, Bt, Bn]``.
    gamma : float
        Adiabatic index.

    Returns
    -------
    ndarray
        Conserved state ``[rho, rho*vr, E, Bt, Bn]``.
    """
    U = np.empty(5)
    U[0], U[1], U[2], U[3], U[4] = _conserved_s(q[0], q[1], q[2], q[3], q[4], gamma)
    return U


@njit(cache=NUMBA_CACHE)
def _fast_speed(q, br, gamma):
    """Return the reduced system's radial fast-mode speed.

    Parameters
    ----------
    q : ndarray
        Primitive MHD state.
    br : float
        Radial magnetic field, retained for the MHD flux interface.
    gamma : float
        Adiabatic index.

    Returns
    -------
    float
        Radial fast-mode speed, in m/s.
    """
    return _fast_speed_s(q[0], q[2], q[3], q[4], gamma)


@njit(cache=NUMBA_CACHE)
def _flux(q, br, gamma):
    """Return the physical flux for a primitive MHD state.

    Parameters
    ----------
    q : ndarray
        Primitive MHD state.
    br : float
        Radial magnetic field at the interface.
    gamma : float
        Adiabatic index.

    Returns
    -------
    ndarray
        Mass, momentum, energy, and transverse induction fluxes.
    """
    F = np.empty(5)
    F[0], F[1], F[2], F[3], F[4] = _flux_s((q[0], q[1], q[2], q[3], q[4]), gamma)
    return F


@njit(cache=NUMBA_CACHE)
def _hll_flux(ql, qr, br, gamma):
    """Compute the two-wave HLL flux across an interface.

    Parameters
    ----------
    ql, qr : ndarray
        Left and right primitive states.
    br : float
        Common divergence-free radial field.
    gamma : float
        Adiabatic index.

    Returns
    -------
    ndarray
        HLL interface flux.
    """
    F = np.empty(5)
    F[0], F[1], F[2], F[3], F[4] = _hll_flux_s(
        (ql[0], ql[1], ql[2], ql[3], ql[4]), (qr[0], qr[1], qr[2], qr[3], qr[4]), gamma)
    return F


# Finite-volume operator and time stepping

@njit(cache=NUMBA_CACHE)
def _rhs_into(U, bc, r, faces, gamma, plm, gravity, result, q, slopes, fluxes):
    """Evaluate the finite-volume spatial operator for the MHD state.

    Parameters
    ----------
    U : ndarray
        Conserved cell states.
    bc : ndarray
        Primitive inner-boundary state.
    r, faces : ndarray
        Cell-centre and interface radii, in m.
    gamma : float
        Adiabatic index.
    plm, gravity : bool
        Whether to use PLM reconstruction and solar gravity.
    result : ndarray
        Time derivative of each conserved cell state, filled in place.
    q, slopes, fluxes : ndarray
        Work arrays.

    Returns
    -------
    None
    """
    nr = len(r)
    for i in range(nr):
        q[i, 0], q[i, 1], q[i, 2], q[i, 3], q[i, 4] = _primitive_s(
            U[i, 0], U[i, 1], U[i, 2], U[i, 3], U[i, 4], gamma)
    slopes[:, :] = 0.0
    if plm:
        for i in range(1, nr - 1):
            for k in range(5):
                slopes[i, k] = _mc_limiter(
                    (q[i, k] - q[i - 1, k]) / (r[i] - r[i - 1]),
                    (q[i + 1, k] - q[i, k]) / (r[i + 1] - r[i]),
                )
            # Fall back locally to PCM if reconstruction is not admissible.
            for face in (faces[i], faces[i + 1]):
                if (q[i, 0] + slopes[i, 0] * (face - r[i]) <= SMALL_RHO
                        or q[i, 2] + slopes[i, 2] * (face - r[i]) <= SMALL_P):
                    slopes[i, :] = 0.0

    for j in range(nr + 1):
        if j == 0:
            ql = (bc[0], bc[1], bc[2], bc[3], bc[4])
            qr = (q[0, 0], q[0, 1], q[0, 2], q[0, 3], q[0, 4])
        elif j == nr:
            # zero-gradient outflow
            ql = (q[nr - 1, 0], q[nr - 1, 1], q[nr - 1, 2], q[nr - 1, 3], q[nr - 1, 4])
            qr = ql
        else:
            dl = faces[j] - r[j - 1]
            dr = faces[j] - r[j]
            ql = (q[j - 1, 0] + slopes[j - 1, 0] * dl, q[j - 1, 1] + slopes[j - 1, 1] * dl,
                  q[j - 1, 2] + slopes[j - 1, 2] * dl, q[j - 1, 3] + slopes[j - 1, 3] * dl,
                  q[j - 1, 4] + slopes[j - 1, 4] * dl)
            qr = (q[j, 0] + slopes[j, 0] * dr, q[j, 1] + slopes[j, 1] * dr,
                  q[j, 2] + slopes[j, 2] * dr, q[j, 3] + slopes[j, 3] * dr,
                  q[j, 4] + slopes[j, 4] * dr)
        (fluxes[j, 0], fluxes[j, 1], fluxes[j, 2],
         fluxes[j, 3], fluxes[j, 4]) = _hll_flux_s(ql, qr, gamma)

    for i in range(nr):
        lo, hi = faces[i], faces[i + 1]
        # Difference-of-powers factorizations avoid cancellation for thin shells.
        w2 = (hi - lo) * (hi * hi + hi * lo + lo * lo) / 3
        w1 = (hi - lo) * (hi + lo) / 2
        for k in range(5):
            if k == 3 or k == 4:  # transverse induction: d(r F)/dr
                result[i, k] = -(hi * fluxes[i + 1, k] - lo * fluxes[i, k]) / w1
            else:
                result[i, k] = -(hi**2 * fluxes[i + 1, k]
                                 - lo**2 * fluxes[i, k]) / w2
        inv_r = w1 / w2
        result[i, 1] += 2 * q[i, 2] * inv_r
        if gravity:
            acceleration = -GM_SUN / r[i]**2
            result[i, 1] += U[i, 0] * acceleration
            result[i, 2] += U[i, 1] * acceleration


@njit(cache=NUMBA_CACHE)
def _rhs(U, bc, r, faces, magnetic_flux, gamma, plm, gravity):
    """Return the finite-volume spatial operator for the MHD state.

    Parameters
    ----------
    U : ndarray
        Conserved cell states.
    bc : ndarray
        Primitive inner-boundary state.
    r, faces : ndarray
        Cell-centre and interface radii, in m.
    magnetic_flux : float
        Constant value of ``r**2 * Br``. Br is force free, so it does not enter.
    gamma : float
        Adiabatic index.
    plm, gravity : bool
        Whether to use PLM reconstruction and solar gravity.

    Returns
    -------
    ndarray
        Time derivative of each conserved cell state.
    """
    nr = len(r)
    result = np.empty((nr, 5))
    _rhs_into(U, bc, r, faces, gamma, plm, gravity, result,
              np.empty((nr, 5)), np.empty((nr, 5)), np.empty((nr + 1, 5)))
    return result


@njit(cache=NUMBA_CACHE)
def _admissible(U, gamma):
    """Check whether every cell has a finite, positive plasma state.

    Parameters
    ----------
    U : ndarray
        Conserved cell states.
    gamma : float
        Adiabatic index.

    Returns
    -------
    bool
        True when all states are physically admissible.
    """
    for i in range(len(U)):
        for k in range(5):
            if not np.isfinite(U[i, k]):
                return False
        if U[i, 0] <= SMALL_RHO:
            return False
        if _primitive_s(U[i, 0], U[i, 1], U[i, 2], U[i, 3], U[i, 4], gamma)[2] <= SMALL_P:
            return False
    return True


@njit(cache=NUMBA_CACHE)
def _mhd_dt(U, br, dx, gamma, cfl):
    """Return the CFL-limited MHD time step.

    Parameters
    ----------
    U : ndarray
        Conserved cell states.
    br, dx : ndarray
        Radial field and cell width for each cell.
    gamma, cfl : float
        Adiabatic index and CFL number.

    Returns
    -------
    float
        Stable time step, in s.
    """
    dt = np.inf
    for i in range(len(U)):
        q = _primitive_s(U[i, 0], U[i, 1], U[i, 2], U[i, 3], U[i, 4], gamma)
        speed = abs(q[1]) + _fast_speed_s(q[0], q[2], q[3], q[4], gamma)
        dt = min(dt, cfl * dx[i] / speed)
    return dt


@njit(cache=NUMBA_CACHE)
def _mhd_boundary(t, bc_times, bc_v, bc_rho, bc_T, bt_mode, bt_x, bt_y, bt_par,
                  bn_mode, bn_x, bn_y, bn_par, q):
    """Fill the primitive inner-boundary state at a model time.

    Parameters
    ----------
    t : float
        Model time, in s.
    bc_times, bc_v, bc_rho, bc_T : ndarray
        Plasma boundary sample times and values in SI units.
    bt_mode, bt_x, bt_y, bt_par, bn_mode, bn_x, bn_y, bn_par
        Tabulated transverse field boundaries, as for ``_eval_boundary``.
    q : ndarray
        Primitive boundary state, filled in place.

    Returns
    -------
    None
    """
    rho = np.interp(t, bc_times, bc_rho)
    vr = np.interp(t, bc_times, bc_v)
    temperature = np.interp(t, bc_times, bc_T)
    bt = _eval_boundary(bt_mode, bt_x, bt_y, bt_par, t)
    bn = _eval_boundary(bn_mode, bn_x, bn_y, bn_par, t)
    if not (np.isfinite(bt) and np.isfinite(bn)):
        raise ValueError('MHD boundary values must be finite')
    q[0] = rho
    q[1] = vr
    q[2] = rho * K_B_SI * temperature / M_P_SI
    q[3] = bt
    q[4] = bn
    for k in range(5):
        if not np.isfinite(q[k]):
            raise ValueError('MHD inflow density and temperature must be positive and finite')
    if q[0] <= SMALL_RHO or q[2] <= SMALL_P:
        raise ValueError('MHD inflow density and temperature must be positive and finite')


@njit(cache=NUMBA_CACHE, nogil=True)
def _run_mhd(U, time, t_grid, bc_times, bc_v, bc_rho, bc_T,
             bt_mode, bt_x, bt_y, bt_par, bn_mode, bn_x, bn_y, bn_par,
             r, faces, dx, min_dx, magnetic_flux, br_cells, gamma, cfl, plm_default, rk2,
             gravity, p_r, p_v, p_active, p_injected, p_inj_times, p_rel_times,
             p_behaviors, p_positive_time, p_step_times, v_out, rho_out, T_out, b_out,
             rec_r, rec_v, rec_t, rec_on):
    """Advance the MHD solution through every output time.

    Each step is limited by the CFL condition, including the changing inflow. Steps that
    give a nonphysical state are retried with PCM and half the time step; negative thermal
    energy is never hidden by clipping it.

    Parameters
    ----------
    U : ndarray
        Conserved cell states, updated in place.
    time : float
        Initial time, in s.
    t_grid : ndarray
        Output times, in s.
    bc_times, bc_v, bc_rho, bc_T : ndarray
        Plasma boundary sample times and values in SI units.
    bt_mode, bt_x, bt_y, bt_par, bn_mode, bn_x, bn_y, bn_par
        Tabulated transverse field boundaries, as for ``_eval_boundary``.
    r, faces, dx : ndarray
        Cell-centre radii, interface radii and cell widths, in m.
    min_dx, magnetic_flux : float
        Smallest cell width and ``r**2 * Br``.
    br_cells : ndarray
        Radial field at each cell centre, in T.
    gamma, cfl : float
        Adiabatic index and CFL number.
    plm_default, rk2, gravity : bool
        PLM reconstruction, RK2 integration and solar gravity switches.
    p_r, p_v, p_active, p_injected : ndarray
        Tracer state, updated in place.
    p_inj_times, p_rel_times, p_behaviors, p_positive_time : ndarray
        Tracer injection and release times and boundary behaviour.
    p_step_times : ndarray
        Fixed tracer-update times, in s.
    v_out, rho_out, T_out, b_out : ndarray
        Plasma and magnetic field output at each output time, filled in place.
    rec_r, rec_v, rec_t, rec_on : ndarray
        Tracer output, filled in place.

    Returns
    -------
    time : float
        Final time, in s.
    rejected : int
        Number of rejected step attempts.
    """
    nr = r.size
    nt = t_grid.size
    face_br = magnetic_flux / faces[:-1]**2
    k1 = np.empty_like(U)
    stage = np.empty_like(U)
    result = np.empty_like(U)
    q = np.empty_like(U)
    slopes = np.empty_like(U)
    fluxes = np.empty((nr + 1, 5))
    bc0 = np.empty(5)
    bc1 = np.empty(5)
    rejected = 0

    p_step_idx = np.searchsorted(p_step_times, time, side='right')
    previous_particle_time = time

    t_idx = 0
    if abs(t_grid[0] - time) < 1e-5:
        _mhd_snapshot(U, gamma, br_cells, time, 0, v_out, rho_out, T_out, b_out,
                      p_r, p_v, p_active, rec_r, rec_v, rec_t, rec_on)
        t_idx = 1

    while t_idx < nt:
        target_time = t_grid[t_idx]
        while time < target_time:
            dt = _mhd_dt(U, face_br, dx, gamma, cfl)
            if time + dt > target_time:
                dt = target_time - time

            plm = plm_default
            accepted = False
            for attempt in range(24):
                _mhd_boundary(time, bc_times, bc_v, bc_rho, bc_T, bt_mode, bt_x, bt_y, bt_par,
                              bn_mode, bn_x, bn_y, bn_par, bc0)
                _mhd_boundary(time + dt, bc_times, bc_v, bc_rho, bc_T, bt_mode, bt_x, bt_y,
                              bt_par, bn_mode, bn_x, bn_y, bn_par, bc1)
                speed = max(
                    abs(bc0[1]) + _fast_speed_s(bc0[0], bc0[2], bc0[3], bc0[4], gamma),
                    abs(bc1[1]) + _fast_speed_s(bc1[0], bc1[2], bc1[3], bc1[4], gamma),
                )
                limit = cfl * min_dx / speed
                if dt > limit * (1 + 1e-12):
                    dt = limit
                    continue
                _rhs_into(U, bc0, r, faces, gamma, plm, gravity, k1, q, slopes, fluxes)
                for i in range(nr):
                    for k in range(5):
                        stage[i, k] = U[i, k] + dt * k1[i, k]
                if _admissible(stage, gamma):
                    if rk2:
                        _rhs_into(stage, bc1, r, faces, gamma, plm, gravity, k1, q, slopes,
                                  fluxes)
                        for i in range(nr):
                            for k in range(5):
                                result[i, k] = 0.5 * (U[i, k] + stage[i, k] + dt * k1[i, k])
                    else:
                        result[:, :] = stage
                    if _admissible(result, gamma):
                        if time + dt == time:
                            raise RuntimeError('MHD timestep underflow')
                        U[:, :] = result
                        time += dt
                        rejected += attempt
                        accepted = True
                        break
                dt *= 0.5
                plm = False
            if not accepted:
                raise RuntimeError('MHD could not find a positive finite update after 24 retries')

            if (p_r.size
                    and p_step_idx < p_step_times.size
                    and p_step_times[p_step_idx] <= time + 1e-7 * max(1.0, abs(time))):
                p_step_idx, previous_particle_time = _catch_up_particles_huxt_batch(
                    p_r, p_v, p_active, p_injected, p_inj_times, p_rel_times,
                    p_behaviors, p_positive_time, U, r, p_step_times,
                    p_step_idx, previous_particle_time, time)

        _mhd_snapshot(U, gamma, br_cells, time, t_idx, v_out, rho_out, T_out, b_out,
                      p_r, p_v, p_active, rec_r, rec_v, rec_t, rec_on)
        t_idx += 1
    return time, rejected


@njit(cache=NUMBA_CACHE)
def _mhd_snapshot(U, gamma, br_cells, time, t_idx, v_out, rho_out, T_out, b_out,
                  p_r, p_v, p_active, rec_r, rec_v, rec_t, rec_on):
    """Store the plasma and magnetic fields and the active tracers at an output time.

    Parameters
    ----------
    U : ndarray
        Conserved cell states.
    gamma : float
        Adiabatic index.
    br_cells : ndarray
        Radial field at each cell centre, in T.
    time : float
        Current time, in s.
    t_idx : int
        Output time index.
    v_out, rho_out, T_out, b_out : ndarray
        Plasma and field output, filled in place.
    p_r, p_v, p_active : ndarray
        Tracer positions, speeds and activity.
    rec_r, rec_v, rec_t, rec_on : ndarray
        Tracer output, filled in place.

    Returns
    -------
    None
    """
    for i in range(len(U)):
        q = _primitive_s(U[i, 0], U[i, 1], U[i, 2], U[i, 3], U[i, 4], gamma)
        v_out[t_idx, i] = q[1]
        rho_out[t_idx, i] = q[0]
        T_out[t_idx, i] = q[2] * M_P_SI / (q[0] * K_B_SI)
        b_out[t_idx, i, 0] = br_cells[i]
        b_out[t_idx, i, 1] = q[3]
        b_out[t_idx, i, 2] = q[4]
    _record_particles(time, t_idx, p_r, p_v, p_active, rec_r, rec_v, rec_t, rec_on)


class MHDSolver(CompressibleSolver):
    """Propagate a radial-flow ideal-MHD state in SI units.

    ``br`` fixes radial magnetic flux. ``boundary`` and ``mhd_init`` accept
    only the transverse magnetic components ``bt`` and ``bn``.
    """

    def __init__(self, r_grid, gamma=1.5, cfl=None, reconstruction='plm',
                 time_integration='rk2', verbose=False, pui=False, br=0.0,
                 boundary=None, gravity=False):
        """Initialise a radial ideal-MHD solver.

        Parameters
        ----------
        r_grid : ndarray
            Radial cell centres, in m.
        gamma : float
            Adiabatic index.
        cfl : float, optional
            CFL number; defaults to 0.35.
        reconstruction, time_integration : str
            Spatial reconstruction and time-integration methods.
        verbose : bool
            Compatibility flag; no terminal output is produced.
        pui : bool
            Unsupported hydro PUI flag; must be False.
        br : float
            Radial field at the inner cell centre, in T.
        boundary : mapping, optional
            Transverse magnetic boundary values: numbers, in T, or the tabulated forms
            from ``time_series_boundary`` and ``periodic_boundary``.
        gravity : bool
            Whether to include solar gravity.

        Returns
        -------
        None
        """
        r_grid = np.asarray(r_grid, dtype=float)
        if (r_grid.ndim != 1 or len(r_grid) < 3 or
                not np.all(np.isfinite(r_grid)) or np.any(np.diff(r_grid) <= 0) or
                r_grid[0] - 0.5 * (r_grid[1] - r_grid[0]) <= 0):
            raise ValueError('MHD requires at least 3 increasing positive shell centres')
        if not np.isfinite(gamma) or gamma <= 1:
            raise ValueError('MHD gamma must exceed 1')
        if (reconstruction not in ('plm', 'pcm')
                or time_integration not in ('rk2', 'euler')):
            raise ValueError('MHD supports pcm/plm reconstruction and euler/rk2 integration')
        if cfl is None:
            cfl = 0.35
        if not np.isfinite(cfl) or not 0 < cfl <= 0.5:
            raise ValueError('MHD CFL must be in (0, 0.5]')
        if pui:
            raise ValueError('The empirical hydro PUI source is not an MHD pickup-ion model')
        if np.ndim(br) != 0 or not np.isfinite(br):
            raise ValueError('br must be a finite scalar: r^2 Br is fixed in space and time')
        super().__init__(r_grid, gamma, cfl, reconstruction=reconstruction,
                         time_integration=time_integration, verbose=verbose)
        self.magnetic_flux = float(br) * self.r[0]**2
        self.br = self.magnetic_flux / self.r**2
        self.gravity = bool(gravity)
        boundary = dict(boundary or {})
        if set(boundary)-{'bt', 'bn'}:
            raise ValueError('MHD boundary keys must be bt and bn; specify br separately')
        self.boundary = {}
        for key in ('bt', 'bn'):
            value = boundary.get(key, 0.0)
            if callable(value):
                raise TypeError('MHD boundary values must be numbers or tabulated boundaries')
            self.boundary[key] = value if isinstance(value, tuple) else constant_boundary(value)
        self.U = np.zeros((self.nr, 5))

    def _set_initial_conditions(self, rho, v, T):
        """Set the initial plasma and magnetic profiles.

        Parameters
        ----------
        rho, v, T : ndarray
            Density, radial velocity, and temperature in SI units.

        Returns
        -------
        None
            ``self.U`` is updated in place.
        """
        q = np.zeros((self.nr, 5))
        q[:, 0], q[:, 1] = rho, v
        q[:, 2] = np.asarray(rho) * K_B_SI * np.asarray(T) / M_P_SI
        for k, key in ((3, 'bt'), (4, 'bn')):
            b = _eval_boundary(*self.boundary[key], self._initial_time)
            if not np.isfinite(b):
                raise ValueError('MHD boundary values must be finite')
            profile = self._mhd_init.get(key, b * self.r[0] / self.r)
            profile = np.asarray(profile, dtype=float)
            if profile.shape != (self.nr,):
                raise ValueError(f'mhd_init[{key!r}] must have shape ({self.nr},)')
            q[:, k] = profile
        if (not np.all(np.isfinite(q)) or np.any(q[:, 0] <= SMALL_RHO)
                or np.any(q[:, 2] <= SMALL_P)):
            raise ValueError('Initial MHD density and pressure must be positive and finite')
        for i in range(self.nr):
            self.U[i] = _conserved(q[i], self.gamma)

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
            ``self.U``, ``self.time`` and ``self.rejected_steps`` are updated.
        """
        v_out, rho_out, T_out = outputs[:3]
        self._b_out = np.zeros((len(t_grid), self.nr, 3))
        self.time, self.rejected_steps = _run_mhd(
            self.U, self.time, t_grid, *boundary, *self.boundary['bt'], *self.boundary['bn'],
            self.r, self.r_int, self.dx, self.min_dx, self.magnetic_flux, self.br,
            self.gamma, self.cfl, self.reconstruction == 'plm',
            self.time_integration == 'rk2', self.gravity, *particles,
            v_out, rho_out, T_out, self._b_out, *outputs[3:])

    def solve(self, t_grid, v_bc, rho_bc, T_bc, *, mhd_init=None, **kwargs):
        """Run the MHD solver over an output-time grid.

        Parameters
        ----------
        t_grid : ndarray
            Strictly increasing output times, in s.
        v_bc, rho_bc, T_bc : float or ndarray
            Radial velocity, density, and temperature boundary conditions, constant or
            sampled at ``bc_times``.
        mhd_init : mapping, optional
            Initial ``bt`` and ``bn`` radial profiles, in T.
        **kwargs
            ``bc_times``, initial plasma profiles and particle options accepted by
            :meth:`CompressibleSolver.solve`.

        Returns
        -------
        dict
            Plasma, magnetic-field, solver, and optional particle results.
        """
        t_grid = np.asarray(t_grid, dtype=float)
        if (t_grid.ndim != 1 or not len(t_grid)
                or not np.all(np.isfinite(t_grid))
                or np.any(np.diff(t_grid) <= 0)):
            raise ValueError('MHD output times must be finite and strictly increasing')
        self._initial_time = t_grid[0]
        self._mhd_init = dict(mhd_init or {})
        if set(self._mhd_init)-{'bt', 'bn'}:
            raise ValueError('mhd_init keys must be bt and bn')
        supplied = [
            kwargs.get(key) is not None
            for key in ('v_init', 'rho_init', 'T_init')
        ]
        if any(supplied) and not all(supplied):
            raise ValueError('Provide all of v_init, rho_init and T_init for a restart')
        if all(supplied):
            for key in ('v_init', 'rho_init', 'T_init'):
                if np.shape(kwargs[key]) != (self.nr,):
                    raise ValueError(f'{key} must have shape ({self.nr},)')
        result = super().solve(t_grid, v_bc, rho_bc, T_bc, **kwargs)
        for i, key in enumerate(MHD_COMPONENTS):
            result[key] = self._b_out[:, :, i]
        result['solver_info'].update(
            physics='ideal-mhd',
            riemann='hll',
            gravity=self.gravity,
            rejected_steps=self.rejected_steps,
        )
        return result
