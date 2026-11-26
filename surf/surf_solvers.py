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


# Equation of state

@njit(cache=NUMBA_CACHE)
def _get_rhoe(p, gamma):
    """Return internal energy density from pressure.

    Parameters
    ----------
    p : float
        Thermal pressure.
    gamma : float
        Adiabatic index.

    Returns
    -------
    float
        Internal energy density.
    """
    return p / (gamma - 1.0)


@njit(cache=NUMBA_CACHE)
def _prim_to_cons(q, gamma):
    """Convert primitive variables to conserved variables.

    Parameters
    ----------
    q : ndarray
        Primitive state ``[rho, v, p]``.
    gamma : float
        Adiabatic index.

    Returns
    -------
    ndarray
        Conserved state ``[rho, rho*v, E]``.
    """
    rho, v, p = q[0], q[1], q[2]
    mom = rho * v
    rhoe = _get_rhoe(p, gamma)
    ener = rhoe + 0.5 * rho * v**2
    return np.array([rho, mom, ener])


# HLLC Riemann solver

@njit(cache=NUMBA_CACHE)
def _riemann_hllc(U_l, U_r, gamma):
    """Return the HLLC interface flux for two conserved states.
    
    Parameters
    ----------
    U_l, U_r : ndarray
        Left and right conserved states ``[rho, rho*v, E]``.
    gamma : float
        Adiabatic index.
        
    Returns
    -------
    ndarray
        Mass, momentum, and energy fluxes.
    """
    # Left state primitives
    rho_l = max(U_l[0], SMALL_RHO)
    v_l = U_l[1] / rho_l
    E_l = U_l[2]
    p_l = max((E_l - 0.5*rho_l*v_l**2) * (gamma - 1.0), SMALL_P)
    c_l = np.sqrt(gamma * p_l / rho_l)
    
    # Right state primitives
    rho_r = max(U_r[0], SMALL_RHO)
    v_r = U_r[1] / rho_r
    E_r = U_r[2]
    p_r = max((E_r - 0.5*rho_r*v_r**2) * (gamma - 1.0), SMALL_P)
    c_r = np.sqrt(gamma * p_r / rho_r)
    
    # Wave speed estimates
    S_l = min(v_l - c_l, v_r - c_r)
    S_r = max(v_l + c_l, v_r + c_r)
    
    F = np.zeros(3)
    
    if S_l >= 0:
        # Supersonic flow to the right
        F[0] = rho_l * v_l
        F[1] = rho_l * v_l * v_l + p_l
        F[2] = (E_l + p_l) * v_l
        return F
    elif S_r <= 0:
        # Supersonic flow to the left
        F[0] = rho_r * v_r
        F[1] = rho_r * v_r * v_r + p_r
        F[2] = (E_r + p_r) * v_r
        return F
    
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
        
        F_l_0 = rho_l * v_l
        F_l_1 = rho_l * v_l * v_l + p_l
        F_l_2 = (E_l + p_l) * v_l
        
        F[0] = F_l_0 + S_l * (U_star_0 - U_l[0])
        F[1] = F_l_1 + S_l * (U_star_1 - U_l[1])
        F[2] = F_l_2 + S_l * (U_star_2 - U_l[2])
    else:
        # Right star state
        factor = rho_r * (S_r - v_r) / (S_r - S_c)
        U_star_0 = factor
        U_star_1 = factor * S_c
        U_star_2 = factor * (E_r/rho_r + (S_c - v_r) * (S_c + p_r/(rho_r*(S_r - v_r))))
        
        F_r_0 = rho_r * v_r
        F_r_1 = rho_r * v_r * v_r + p_r
        F_r_2 = (E_r + p_r) * v_r
        
        F[0] = F_r_0 + S_r * (U_star_0 - U_r[0])
        F[1] = F_r_1 + S_r * (U_star_1 - U_r[1])
        F[2] = F_r_2 + S_r * (U_star_2 - U_r[2])
    
    return F


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
def _compute_fluxes_pcm(U, nr, gamma):
    """Compute first-order piecewise-constant interface fluxes.

    Parameters
    ----------
    U : ndarray
        Conserved cell states.
    nr : int
        Number of radial cells.
    gamma : float
        Adiabatic index.

    Returns
    -------
    ndarray
        HLLC flux at each cell interface.
    """
    fluxes = np.zeros((nr + 1, 3))
    
    for i in range(nr + 1):
        if i == 0:
            U_l = U[0]
            U_r = U[0]
        elif i == nr:
            U_l = U[nr-1]
            U_r = U[nr-1]
        else:
            U_l = U[i-1]
            U_r = U[i]
        
        fluxes[i] = _riemann_hllc(U_l, U_r, gamma)
    
    return fluxes


@njit(cache=NUMBA_CACHE)
def _compute_fluxes_plm(U, nr, gamma):
    """Compute second-order piecewise-linear interface fluxes.

    Parameters
    ----------
    U : ndarray
        Conserved cell states.
    nr : int
        Number of radial cells.
    gamma : float
        Adiabatic index.

    Returns
    -------
    ndarray
        Reconstructed HLLC flux at each cell interface.
    """
    fluxes = np.zeros((nr + 1, 3))
    
    # Convert all cells to primitives [rho, v, p]
    q_all = np.zeros((nr, 3))
    q_valid = np.ones(nr, dtype=np.int32)
    for j in range(nr):
        rho = max(U[j, 0], SMALL_RHO)
        v = U[j, 1] / rho
        eint = (U[j, 2] - 0.5 * rho * v * v) / rho
        p = max(rho * eint * (gamma - 1.0), SMALL_P)
        q_all[j, 0] = rho
        q_all[j, 1] = v
        q_all[j, 2] = p
        if rho <= 0 or p <= 0:
            q_valid[j] = 0
    
    for i in range(nr + 1):
        if i == 0:
            U_l = U[0]
            U_r = U[0]
        elif i == nr:
            # Outer boundary - 2nd order extrapolation
            if nr >= 2:
                q_last = q_all[nr-1]
                q_prev = q_all[nr-2]
                q_face = np.zeros(3)
                q_face[0] = max(q_last[0] + 0.5 * (q_last[0] - q_prev[0]), SMALL_RHO)
                q_face[1] = q_last[1] + 0.5 * (q_last[1] - q_prev[1])
                q_face[2] = max(q_last[2] + 0.5 * (q_last[2] - q_prev[2]), SMALL_P)
                U_face = _prim_to_cons(q_face, gamma)
                U_l = U_face
                U_r = U_face
            else:
                U_l = U[nr-1]
                U_r = U[nr-1]
        else:
            idx = i - 1  # left cell of interface
            
            if idx >= nr - 1:
                U_l = U[idx].copy()
                U_r = U[idx].copy()
            elif idx == 0:
                # Inner boundary special case
                if nr < 3 or q_valid[0] == 0 or q_valid[1] == 0 or q_valid[2] == 0:
                    U_l = U[0].copy()
                    U_r = U[1].copy()
                else:
                    q_L = np.zeros(3)
                    q_R = np.zeros(3)
                    for n in range(3):
                        slope_0 = q_all[1, n] - q_all[0, n]
                        slope_L = q_all[1, n] - q_all[0, n]
                        slope_R = q_all[2, n] - q_all[1, n]
                        dq_1 = _mc_limiter(slope_L, slope_R)
                        q_L[n] = q_all[0, n] + 0.5 * slope_0
                        q_R[n] = q_all[1, n] - 0.5 * dq_1
                    q_L[0] = max(q_L[0], SMALL_RHO); q_L[2] = max(q_L[2], SMALL_P)
                    q_R[0] = max(q_R[0], SMALL_RHO); q_R[2] = max(q_R[2], SMALL_P)
                    U_l = _prim_to_cons(q_L, gamma)
                    U_r = _prim_to_cons(q_R, gamma)
            else:
                # Standard internal interface
                if q_valid[idx-1] == 0 or q_valid[idx] == 0 or q_valid[idx+1] == 0:
                    U_l = U[idx].copy()
                    U_r = U[idx+1].copy()
                else:
                    dq_l = np.zeros(3)
                    dq_r = np.zeros(3)
                    for n in range(3):
                        slope_l = q_all[idx, n] - q_all[idx-1, n]
                        slope_r = q_all[idx+1, n] - q_all[idx, n]
                        dq_l[n] = _mc_limiter(slope_l, slope_r)
                        
                        if idx < nr - 2:
                            slope_l2 = q_all[idx+1, n] - q_all[idx, n]
                            slope_r2 = q_all[idx+2, n] - q_all[idx+1, n]
                            dq_r[n] = _mc_limiter(slope_l2, slope_r2)
                    
                    q_L = q_all[idx] + 0.5 * dq_l
                    q_R = q_all[idx+1] - 0.5 * dq_r
                    q_L[0] = max(q_L[0], SMALL_RHO); q_L[2] = max(q_L[2], SMALL_P)
                    q_R[0] = max(q_R[0], SMALL_RHO); q_R[2] = max(q_R[2], SMALL_P)
                    U_l = _prim_to_cons(q_L, gamma)
                    U_r = _prim_to_cons(q_R, gamma)
        
        fluxes[i] = _riemann_hllc(U_l, U_r, gamma)
    
    return fluxes


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
def _extract_snapshot(U, nr, gamma, M_P_val, K_B_val):
    """Extract primitive fields from a conserved state.

    Parameters
    ----------
    U : ndarray
        Conserved cell states.
    nr : int
        Number of radial cells.
    gamma : float
        Adiabatic index.
    M_P_val, K_B_val : float
        Proton mass and Boltzmann constant in SI units.

    Returns
    -------
    tuple of ndarray
        Velocity, density, and temperature profiles.
    """
    v_out = np.zeros(nr)
    rho_out = np.zeros(nr)
    T_out = np.zeros(nr)
    for i in range(nr):
        rho = max(U[i, 0], SMALL_RHO)
        v = U[i, 1] / rho
        eint = (U[i, 2] - 0.5 * rho * v * v) / rho
        p = max(rho * eint * (gamma - 1.0), SMALL_P)
        v_out[i] = v
        rho_out[i] = rho
        T_out[i] = p * M_P_val / (rho * K_B_val)
    return v_out, rho_out, T_out


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
    ndarray
        Updated conserved states.
    """
    if not enabled:
        return U
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
    return U


@njit(cache=NUMBA_CACHE)
def _step_euler(U, U_bc, nr, A, V, r, dt, gamma, use_plm, pui):
    """Take an Euler step including spherical geometry and PUI forcing.

    Parameters
    ----------
    U, U_bc : ndarray
        Conserved cell states and inner-boundary state.
    nr : int
        Number of radial cells.
    A, V, r : ndarray
        Interface areas, cell volumes, and cell-centre radii.
    dt, gamma : float
        Time step and adiabatic index.
    use_plm, pui : bool
        Whether to use PLM reconstruction and PUI forcing.

    Returns
    -------
    ndarray
        Conserved state after one Euler step.
    """
    if use_plm:
        fluxes = _compute_fluxes_plm(U, nr, gamma)
    else:
        fluxes = _compute_fluxes_pcm(U, nr, gamma)
    
    U_new = np.zeros_like(U)
    
    for i in range(nr):
        if V[i] <= 0:
            U_new[i] = U[i]
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
            U_new[i] = U[i]
    
    U_new[0] = U_bc
    return _apply_pui_source(U_new, r, nr, dt, pui)


@njit(cache=NUMBA_CACHE)
def _step_rk2(U, U_bc, nr, A, V, r, dt, gamma, use_plm, pui):
    """Take a second-order Heun step.

    Parameters
    ----------
    U, U_bc : ndarray
        Conserved cell states and inner-boundary state.
    nr : int
        Number of radial cells.
    A, V, r : ndarray
        Interface areas, cell volumes, and cell-centre radii.
    dt, gamma : float
        Time step and adiabatic index.
    use_plm, pui : bool
        Whether to use PLM reconstruction and PUI forcing.

    Returns
    -------
    ndarray
        Conserved state after one RK2 step.
    """
    U1 = _step_euler(U, U_bc, nr, A, V, r, dt, gamma, use_plm, pui)
    U2 = _step_euler(U1, U_bc, nr, A, V, r, dt, gamma, use_plm, pui)
    
    U_new = np.zeros_like(U)
    for i in range(nr):
        for j in range(3):
            U_new[i, j] = 0.5 * (U[i, j] + U2[i, j])
    
    U_new[0] = U_bc
    return U_new


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
        self.r = r_grid.copy()
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
            self.U[i] = _prim_to_cons(np.array([rho[i], v[i], p]), self.gamma)
    
    def _get_dt(self):
        """Return the CFL-limited time step.

        Returns
        -------
        float
            Stable time step, in s.
        """
        return _get_dt(self.U, self.nr, self.min_dx, self.gamma, self.cfl)

    def _snapshot(self):
        """Return the current primitive plasma fields.

        Returns
        -------
        tuple of ndarray
            Velocity, density, and temperature profiles.
        """
        return _extract_snapshot(self.U, self.nr, self.gamma, M_P_SI, K_B_SI)
    
    def _step(self, dt, bc_func):
        """Advance the solution by one time step.

        Parameters
        ----------
        dt : float
            Time step, in s.
        bc_func : callable
            Function returning ``(rho, v, T)`` at a requested time.

        Returns
        -------
        ndarray
            Updated conserved state.
        """
        rho_bc, v_bc, T_bc = bc_func(self.time + dt)
        p_bc = rho_bc * K_B_SI * T_bc / M_P_SI
        U_bc = _prim_to_cons(np.array([rho_bc, v_bc, p_bc]), self.gamma)
        
        use_plm = (self.reconstruction == 'plm')
        
        if self.time_integration == 'rk2':
            self.U = _step_rk2(self.U, U_bc, self.nr, self.A, self.V, self.r,
                               dt, self.gamma, use_plm, self.pui)
        else:
            self.U = _step_euler(self.U, U_bc, self.nr, self.A, self.V, self.r,
                                 dt, self.gamma, use_plm, self.pui)
        
        self.time += dt
        return self.U
    
    def solve(self, t_grid, v_bc_func, rho_bc_func, T_bc_func,
              num_particles=None, particle_injection_rate=None, particle_release_rate=None,
              particle_initial_positions=None,
              v_init=None, rho_init=None, T_init=None,
              particle_step_times=None):
        """Run the solver and optional tracers over an output-time grid.
        
        Parameters
        ----------
        t_grid : ndarray
            Output times, in s.
        v_bc_func : callable
            Velocity boundary condition, in m/s.
        rho_bc_func : callable
            Density boundary condition, in kg/m³.
        T_bc_func : callable
            Temperature boundary condition, in K.
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
        # Initialize state: use provided profiles or power-law scaling
        if v_init is not None and rho_init is not None and T_init is not None:
            self._set_initial_conditions(rho_init, v_init, T_init)
        else:
            v0 = v_bc_func(t_grid[0])
            rho0 = rho_bc_func(t_grid[0])
            T0 = T_bc_func(t_grid[0])
            
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
        
        # Output arrays
        nt = len(t_grid)
        v_out = np.zeros((nt, self.nr))
        rho_out = np.zeros((nt, self.nr))
        T_out = np.zeros((nt, self.nr))
        
        # Particle tracking setup
        particles_enabled = False
        particle_groups = {}
        
        if num_particles:
            particles_enabled = True
            v_initial = self.U[:, 1] / self.U[:, 0]
            for group_name, n_p in num_particles.items():
                inj_times = particle_injection_rate[group_name]
                rel_times = particle_release_rate[group_name] \
                    if particle_release_rate and group_name in particle_release_rate else inj_times
                behavior = 1 if 'cme' in group_name.lower() else 0
                
                initial_positions = (
                    particle_initial_positions.get(group_name, [])
                    if particle_initial_positions else [])
                initial_positions = list(initial_positions)
                initial_velocities = [
                    np.interp(r_p, self.r, v_initial)
                    for r_p in initial_positions]
                particle_groups[group_name] = {
                    'n_particles': n_p,
                    'injection_times': inj_times,
                    'release_times': rel_times,
                    'r': [[r_p] for r_p in initial_positions],
                    'v': [[v_p] for v_p in initial_velocities],
                    't': [[t_grid[0]] for _ in initial_positions],
                    't_inject': [t_grid[0] for _ in initial_positions],
                    'active': [True for _ in initial_positions],
                    'r_current': initial_positions.copy(),
                    'v_current': initial_velocities.copy(),
                    'initial_velocities': initial_velocities.copy(),
                    'particles_injected': len(initial_positions),
                    'behavior': behavior,
                    'huxt_cme': 'cme' in group_name.lower(),
                }

        # Pack every named group into flat arrays so one compiled call updates
        # CME, HCS, and streakline tracers on each internal step.
        cme_r_initial = []
        cme_v_initial = []
        cme_active_initial = []
        cme_injected_initial = []
        cme_injection_times = []
        cme_release_times = []
        particle_behaviors = []
        require_positive_time = []
        for group in particle_groups.values():
            n_group = group['n_particles']
            initial_positions = list(group.get('r_current', []))
            initial_velocities = list(group.get('initial_velocities', []))
            group['cme_batch_start'] = len(cme_r_initial)
            for particle_id in range(n_group):
                has_initial = particle_id < len(initial_positions)
                cme_r_initial.append(
                    initial_positions[particle_id] if has_initial else np.nan)
                cme_v_initial.append(
                    initial_velocities[particle_id] if has_initial else np.nan)
                cme_active_initial.append(has_initial)
                cme_injected_initial.append(has_initial)
                cme_injection_times.append(group['injection_times'][particle_id])
                cme_release_times.append(group['release_times'][particle_id])
                particle_behaviors.append(group['behavior'])
                require_positive_time.append(group['huxt_cme'])
            group['r'] = [[] for _ in range(n_group)]
            group['v'] = [[] for _ in range(n_group)]
            group['t'] = [[] for _ in range(n_group)]
            group['t_inject'] = list(group['injection_times'])

        cme_r_current = np.asarray(cme_r_initial, dtype=float)
        cme_v_current = np.asarray(cme_v_initial, dtype=float)
        cme_active = np.asarray(cme_active_initial, dtype=bool)
        cme_injected = np.asarray(cme_injected_initial, dtype=bool)
        cme_injection_times = np.asarray(cme_injection_times, dtype=float)
        cme_release_times = np.asarray(cme_release_times, dtype=float)
        particle_behaviors = np.asarray(particle_behaviors, dtype=np.int64)
        require_positive_time = np.asarray(require_positive_time, dtype=bool)
        if particle_step_times is None:
            particle_step_times = np.asarray(t_grid, dtype=float)
        else:
            particle_step_times = np.asarray(particle_step_times, dtype=float)
        particle_step_idx = int(np.searchsorted(
            particle_step_times, self.time, side='right'))
        previous_particle_time = self.time
        
        # Time loop
        t_idx = 0
        
        # Save first snapshot
        if abs(t_grid[0] - self.time) < 1e-5:
            v_snap, rho_snap, T_snap = self._snapshot()
            v_out[0, :] = v_snap
            rho_out[0, :] = rho_snap
            T_out[0, :] = T_snap
            for group in particle_groups.values():
                batch_start = group['cme_batch_start']
                for particle_id in range(group['n_particles']):
                    batch_id = batch_start + particle_id
                    if cme_active[batch_id]:
                        group['r'][particle_id].append(cme_r_current[batch_id])
                        group['v'][particle_id].append(cme_v_current[batch_id])
                        group['t'][particle_id].append(self.time)
            t_idx = 1
        
        while t_idx < nt:
            target_time = t_grid[t_idx]
            
            while self.time < target_time:
                dt = self._get_dt()
                if self.time + dt > target_time:
                    dt = target_time - self.time
                self._step(dt, lambda t: (rho_bc_func(t), v_bc_func(t), T_bc_func(t)))

                if (cme_r_current.size
                        and particle_step_idx < particle_step_times.size
                        and particle_step_times[particle_step_idx] <= self.time
                        + 1e-7 * max(1.0, abs(self.time))):
                    particle_step_idx, previous_particle_time = (
                        _catch_up_particles_huxt_batch(
                        cme_r_current, cme_v_current, cme_active,
                        cme_injected, cme_injection_times, cme_release_times,
                        particle_behaviors, require_positive_time,
                        self.U, self.r, particle_step_times,
                        particle_step_idx, previous_particle_time,
                        self.time))
                
            
            # Save snapshot
            v_snap, rho_snap, T_snap = self._snapshot()
            v_out[t_idx, :] = v_snap
            rho_out[t_idx, :] = rho_snap
            T_out[t_idx, :] = T_snap
            for group in particle_groups.values():
                batch_start = group['cme_batch_start']
                for particle_id in range(group['n_particles']):
                    batch_id = batch_start + particle_id
                    if cme_active[batch_id]:
                        group['r'][particle_id].append(cme_r_current[batch_id])
                        group['v'][particle_id].append(cme_v_current[batch_id])
                        group['t'][particle_id].append(self.time)
            t_idx += 1

        # Record each group's final tracer state
        if particles_enabled:
            for group in particle_groups.values():
                batch_start = group['cme_batch_start']
                batch_stop = batch_start + group['n_particles']
                group['active'] = list(cme_active[batch_start:batch_stop])
                group['particles_injected'] = int(np.sum(
                    cme_injected[batch_start:batch_stop]))

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
        
        # Format particle data
        if particles_enabled:
            for name, g in particle_groups.items():
                if len(g['r']) > 0:
                    max_len = max(len(traj) for traj in g['r'])
                    n_p = len(g['r'])
                    
                    r_arr = np.full((n_p, max_len), np.nan)
                    v_arr = np.full((n_p, max_len), np.nan)
                    t_arr = np.full((n_p, max_len), np.nan)
                    
                    for i in range(n_p):
                        traj_len = len(g['r'][i])
                        r_arr[i, :traj_len] = g['r'][i]
                        v_arr[i, :traj_len] = g['v'][i]
                        t_arr[i, :traj_len] = g['t'][i]
                    
                    g['r'] = r_arr
                    g['v'] = v_arr
                    g['t'] = t_arr
                    g['t_inject'] = np.array(g['t_inject'])
                    g['active'] = np.array(g['active'])
            
            results['particles'] = {'groups': particle_groups}
        
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
    
    v_bc_func = lambda t: v_bc
    rho_bc_func = lambda t: rho_bc
    T_bc_func = lambda t: T_bc
    
    results = {}
    
    print(f"Running reference: {reference}")
    solver = create_solver(r_grid, gamma=gamma, method=reference, verbose=False)
    t0 = time_module.time()
    ref_result = solver.solve(t_grid, v_bc_func, rho_bc_func, T_bc_func)
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
        result = solver.solve(t_grid, v_bc_func, rho_bc_func, T_bc_func)
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
