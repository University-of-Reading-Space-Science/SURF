"""Native radial-flow ideal MHD in an expanding spherical flux tube.

The plasma velocity is strictly radial. Radial magnetic flux is prescribed as
``r**2 Br = constant``. The evolved SI state is ``[rho, rho*vr, E, Bt, Bn]``;
there are no transverse momentum or transverse velocity degrees of freedom.
"""

import numpy as np
from numba import njit

from surf.surf_solvers import (
    CompressibleSolver, K_B_SI, M_P_SI, NUMBA_CACHE, SMALL_RHO, SMALL_P,
    _mc_limiter,
)

MU0 = 4e-7 * np.pi
GM_SUN = 1.32712440018e20  # m^3 s^-2
MHD_COMPONENTS = ('br', 'bt', 'bn')


def normalize_boundary(boundary, nlon_full, nlon):
    """Validate SURF inputs and convert quantities to SI; never infer B from polarity."""
    from astropy import units as u
    boundary = dict(boundary or {})
    unknown = set(boundary)-set(MHD_COMPONENTS)-{'time'}
    if unknown:
        raise ValueError(f'Unknown mhd_boundary keys: {sorted(unknown)}')
    result = {}
    for key, value in boundary.items():
        unit = u.s if key == 'time' else u.T
        value = np.asarray(value.to_value(unit) if isinstance(value, u.Quantity)
                           else value, dtype=float)
        if not np.all(np.isfinite(value)):
            raise ValueError(f'mhd_boundary[{key!r}] must be finite')
        result[key] = value.copy()
    times = result.get('time')
    if times is not None and (times.ndim != 1 or len(times) < 2 or np.any(np.diff(times) <= 0)):
        raise ValueError('mhd_boundary time must contain at least two increasing seconds')
    for key in MHD_COMPONENTS:
        value = result.setdefault(key, np.asarray(0.0))
        if key == 'br' and value.ndim != 0:
            raise ValueError('br must be scalar and time independent in strict radial MHD')
        if value.ndim == 0:
            continue
        shapes = [(nlon_full,)] if times is None else [(len(times),), (len(times), nlon)]
        if value.shape not in shapes:
            raise ValueError(f'mhd_boundary[{key!r}] needs a scalar or shape in {shapes}')
    return result


def boundary_for_longitude(model, i):
    """Build local SI boundary functions on SURF's time/longitude convention."""
    config = model.mhd_boundary
    offset = getattr(model, '_mhd_time_offset', 0.0)
    boundary = {}
    for key in ('bt', 'bn'):
        value = config[key]
        if value.ndim == 0:
            boundary[key] = float(value)
        elif 'time' in config:
            series = value if value.ndim == 1 else value[:, i]
            boundary[key] = lambda t, series=series: float(np.interp(t+offset, config['time'], series))
        else:
            # Carrington bin centres, same convention as v_boundary. Vectors
            # are inertial RTN components; frame only changes longitude mapping.
            lons = (np.arange(len(value))+0.5)*2*np.pi/len(value)
            lon0 = model.cr_lon_init.to_value('rad') + model.lon[i].to_value('rad')
            omega = 2*np.pi/model.rotation_period.to_value('s')
            boundary[key] = lambda t, value=value, lons=lons: float(np.interp(
                lon0-omega*(t+offset), lons, value, period=2*np.pi))
    return dict(br=float(config['br']), boundary=boundary, gravity=model.mhd_gravity)


@njit(cache=NUMBA_CACHE)
def _primitive(U, gamma):
    """Primitive order: rho, vr, p, Bt, Bn; no pressure clipping."""
    q = U.copy()
    q[1] = U[1] / U[0]
    kinetic = 0.5 * U[1]**2 / U[0]
    q[2] = (gamma - 1) * (U[2] - kinetic - (U[3]**2 + U[4]**2)/(2*MU0))
    return q


@njit(cache=NUMBA_CACHE)
def _conserved(q, gamma):
    U = q.copy()
    U[1] *= q[0]
    U[2] = q[2]/(gamma-1) + 0.5*q[0]*q[1]**2 + (q[3]**2+q[4]**2)/(2*MU0)
    return U


@njit(cache=NUMBA_CACHE)
def _fast_speed(q, br, gamma):
    a2 = gamma*q[2]/q[0]
    bt2 = (q[3]**2+q[4]**2)/(MU0*q[0])
    # With transverse momentum removed, force-free Br has no radial wave
    # characteristic. Only gas and transverse magnetic pressure contribute.
    return np.sqrt(a2 + bt2)


@njit(cache=NUMBA_CACHE)
def _flux(q, br, gamma):
    U = _conserved(q, gamma)
    pt = q[2] + (q[3]**2+q[4]**2)/(2*MU0)
    F = np.empty(5)
    F[0] = U[1]
    # The analytic radial background stress cancels its geometric source.
    F[1] = U[1]*q[1] + pt
    F[2] = (U[2]+pt)*q[1]
    F[3] = q[1]*q[3]
    F[4] = q[1]*q[4]
    return F


@njit(cache=NUMBA_CACHE)
def _hll_flux(ql, qr, br, gamma):
    """Two-wave HLL flux with a common divergence-free normal field."""
    cl = _fast_speed(ql, br, gamma)
    cr = _fast_speed(qr, br, gamma)
    sl = min(ql[1]-cl, qr[1]-cr)
    sr = max(ql[1]+cl, qr[1]+cr)
    fl, fr = _flux(ql, br, gamma), _flux(qr, br, gamma)
    if sl >= 0:
        return fl
    if sr <= 0:
        return fr
    return (sr*fl-sl*fr+sl*sr*(_conserved(qr, gamma)-_conserved(ql, gamma)))/(sr-sl)


@njit(cache=NUMBA_CACHE)
def _rhs(U, bc, r, faces, magnetic_flux, gamma, plm, gravity):
    nr = len(r)
    q = np.empty_like(U)
    slopes = np.zeros_like(U)
    for i in range(nr):
        q[i] = _primitive(U[i], gamma)
    if plm:
        for i in range(1, nr-1):
            for k in range(5):
                slopes[i, k] = _mc_limiter(
                    (q[i, k]-q[i-1, k])/(r[i]-r[i-1]),
                    (q[i+1, k]-q[i, k])/(r[i+1]-r[i]))
            # Fall back locally to PCM if reconstruction is not admissible.
            for face in (faces[i], faces[i+1]):
                if (q[i, 0]+slopes[i, 0]*(face-r[i]) <= SMALL_RHO or
                        q[i, 2]+slopes[i, 2]*(face-r[i]) <= SMALL_P):
                    slopes[i, :] = 0.0
    fluxes = np.empty((nr+1, 5))
    for j in range(nr+1):
        if j == 0:
            ql, qr = bc, q[0]
        elif j == nr:
            ql, qr = q[-1], q[-1]  # zero-gradient outflow
        else:
            ql = q[j-1] + slopes[j-1]*(faces[j]-r[j-1])
            qr = q[j] + slopes[j]*(faces[j]-r[j])
        fluxes[j] = _hll_flux(ql, qr, magnetic_flux/faces[j]**2, gamma)
    result = np.empty_like(U)
    for i in range(nr):
        lo, hi = faces[i], faces[i+1]
        # Difference-of-powers factorizations avoid cancellation for thin shells.
        w2 = (hi-lo)*(hi*hi+hi*lo+lo*lo)/3
        w1 = (hi-lo)*(hi+lo)/2
        for k in range(5):
            if k == 3 or k == 4:  # transverse induction: d(r F)/dr
                result[i, k] = -(hi*fluxes[i+1, k]-lo*fluxes[i, k])/w1
            else:
                result[i, k] = -(hi**2*fluxes[i+1, k]-lo**2*fluxes[i, k])/w2
        inv_r = w1/w2
        result[i, 1] += 2*q[i, 2]*inv_r
        if gravity:
            acceleration = -GM_SUN/r[i]**2
            result[i, 1] += U[i, 0]*acceleration
            result[i, 2] += U[i, 1]*acceleration
    return result


@njit(cache=NUMBA_CACHE)
def _admissible(U, gamma):
    for i in range(len(U)):
        if not np.all(np.isfinite(U[i])) or U[i, 0] <= SMALL_RHO:
            return False
        if _primitive(U[i], gamma)[2] <= SMALL_P:
            return False
    return True


@njit(cache=NUMBA_CACHE)
def _mhd_dt(U, br, dx, gamma, cfl):
    dt = np.inf
    for i in range(len(U)):
        q = _primitive(U[i], gamma)
        dt = min(dt, cfl*dx[i]/(abs(q[1])+_fast_speed(q, br[i], gamma)))
    return dt


class MHDSolver(CompressibleSolver):
    """SI radial-flow MHD propagation solver.

    ``br`` fixes radial magnetic flux. ``boundary`` and ``mhd_init`` accept
    only the two transverse magnetic components ``bt`` and ``bn``.
    """

    def __init__(self, r_grid, gamma=1.5, cfl=None, reconstruction='plm',
                 time_integration='rk2', verbose=False, pui=False, br=0.0,
                 boundary=None, gravity=False):
        r_grid = np.asarray(r_grid, dtype=float)
        if (r_grid.ndim != 1 or len(r_grid) < 3 or
                not np.all(np.isfinite(r_grid)) or np.any(np.diff(r_grid) <= 0) or
                r_grid[0]-0.5*(r_grid[1]-r_grid[0]) <= 0):
            raise ValueError('MHD requires at least 3 increasing positive shell centres')
        if not np.isfinite(gamma) or gamma <= 1:
            raise ValueError('MHD gamma must exceed 1')
        if reconstruction not in ('plm', 'pcm') or time_integration not in ('rk2', 'euler'):
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
        self.magnetic_flux = float(br)*self.r[0]**2
        self.br = self.magnetic_flux/self.r**2
        self.gravity = bool(gravity)
        self.boundary = dict(boundary or {})
        if set(self.boundary)-{'bt', 'bn'}:
            raise ValueError('MHD boundary keys must be bt and bn; specify br separately')
        self.U = np.zeros((self.nr, 5))

    def _boundary_values(self, t):
        values = {}
        for key in ('bt', 'bn'):
            value = self.boundary.get(key, 0.0)
            values[key] = float(value(t) if callable(value) else value)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError('MHD boundary values must be finite')
        return values

    def _set_initial_conditions(self, rho, v, T):
        b = self._boundary_values(self._initial_time)
        q = np.zeros((self.nr, 5))
        q[:, 0], q[:, 1] = rho, v
        q[:, 2] = np.asarray(rho)*K_B_SI*np.asarray(T)/M_P_SI
        for k, key in ((3, 'bt'), (4, 'bn')):
            profile = self._mhd_init.get(key, b[key]*self.r[0]/self.r)
            profile = np.asarray(profile, dtype=float)
            if profile.shape != (self.nr,):
                raise ValueError(f'mhd_init[{key!r}] must have shape ({self.nr},)')
            q[:, k] = profile
        if (not np.all(np.isfinite(q)) or np.any(q[:, 0] <= SMALL_RHO)
                or np.any(q[:, 2] <= SMALL_P)):
            raise ValueError('Initial MHD density and pressure must be positive and finite')
        for i in range(self.nr):
            self.U[i] = _conserved(q[i], self.gamma)

    def _get_dt(self):
        # Br is strongest at each shell's inner face, especially on coarse grids.
        face_br = self.magnetic_flux/self.r_int[:-1]**2
        return _mhd_dt(self.U, face_br, self.dx, self.gamma, self.cfl)

    def _bc(self, t, bc_func):
        rho, vr, temperature = bc_func(t)
        b = self._boundary_values(t)
        q = np.array([rho, vr, rho*K_B_SI*temperature/M_P_SI,
                      b['bt'], b['bn']], dtype=float)
        if not np.all(np.isfinite(q)) or q[0] <= SMALL_RHO or q[2] <= SMALL_P:
            raise ValueError('MHD inflow density and temperature must be positive and finite')
        return q

    def _step(self, dt, bc_func):
        # Include changing inflow in the CFL limit. Reject nonphysical stages,
        # then retry conservatively with PCM and a smaller timestep; never hide
        # negative thermal energy by clipping it in the temperature output.
        plm = self.reconstruction == 'plm'
        for attempt in range(24):
            bc0 = self._bc(self.time, bc_func)
            bc1 = self._bc(self.time+dt, bc_func)
            br_face = self.magnetic_flux/self.r_int[0]**2
            speed = max(abs(bc0[1])+_fast_speed(bc0, br_face, self.gamma),
                        abs(bc1[1])+_fast_speed(bc1, br_face, self.gamma))
            limit = self.cfl*self.min_dx/speed
            if dt > limit*(1+1e-12):
                dt = limit
                continue
            args = (self.r, self.r_int, self.magnetic_flux, self.gamma, plm, self.gravity)
            stage = self.U + dt*_rhs(self.U, bc0, *args)
            if _admissible(stage, self.gamma):
                if self.time_integration == 'rk2':
                    result = 0.5*(self.U+stage+dt*_rhs(stage, bc1, *args))
                else:
                    result = stage
                if _admissible(result, self.gamma):
                    if self.time+dt == self.time:
                        raise RuntimeError('MHD timestep underflow')
                    self.U = result
                    self.time += dt
                    self.rejected_steps += attempt
                    return self.U
            dt *= 0.5
            plm = False
        raise RuntimeError('MHD could not find a positive finite update after 24 retries')

    def _snapshot(self):
        q = np.array([_primitive(row, self.gamma) for row in self.U])
        self._extra_snapshots.append(np.column_stack((self.br, q[:, 3:5])))
        return q[:, 1], q[:, 0], q[:, 2]*M_P_SI/(q[:, 0]*K_B_SI)

    def solve(self, t_grid, v_bc_func, rho_bc_func, T_bc_func, *, mhd_init=None, **kwargs):
        t_grid = np.asarray(t_grid, dtype=float)
        if (t_grid.ndim != 1 or not len(t_grid) or not np.all(np.isfinite(t_grid))
                or np.any(np.diff(t_grid) <= 0)):
            raise ValueError('MHD output times must be finite and strictly increasing')
        self._initial_time = t_grid[0]
        self._mhd_init = dict(mhd_init or {})
        if set(self._mhd_init)-{'bt', 'bn'}:
            raise ValueError('mhd_init keys must be bt and bn')
        supplied = [kwargs.get(key) is not None for key in ('v_init', 'rho_init', 'T_init')]
        if any(supplied) and not all(supplied):
            raise ValueError('Provide all of v_init, rho_init and T_init for a restart')
        if all(supplied):
            for key in ('v_init', 'rho_init', 'T_init'):
                if np.shape(kwargs[key]) != (self.nr,):
                    raise ValueError(f'{key} must have shape ({self.nr},)')
        self._extra_snapshots = []
        self.rejected_steps = 0
        result = super().solve(t_grid, v_bc_func, rho_bc_func, T_bc_func, **kwargs)
        extra = np.asarray(self._extra_snapshots)
        for i, key in enumerate(MHD_COMPONENTS):
            result[key] = extra[:, :, i]
        result['solver_info'].update(physics='ideal-mhd', riemann='hll',
                                     gravity=self.gravity, rejected_steps=self.rejected_steps)
        return result
