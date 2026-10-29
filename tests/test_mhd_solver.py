"""Physical verification of the native solver and its SURF integration."""
import numpy as np
import pytest
from astropy import units as u
from astropy.utils import iers

from surf import surf as s
from surf.surf_solvers import K_B_SI, M_P_SI, create_solver
from surf.surf_mhd import (
    MU0, GM_SUN, MHDSolver, _conserved, _primitive, _flux, _hll_flux,
    _fast_speed, _rhs, normalize_boundary,
)


@pytest.fixture(autouse=True)
def no_iers_download():
    with iers.conf.set_temp('auto_download', False):
        yield


def test_mhd_energy_and_flux_in_si_units():
    q = np.array([1e-18, 4e5, 2e-11, -2e-8, 1e-8])
    br, gamma = 3e-8, 1.5
    conserved = _conserved(q, gamma)
    np.testing.assert_allclose(_primitive(conserved, gamma), q, rtol=1e-12)
    flux = _hll_flux(q, q, br, gamma)
    np.testing.assert_allclose(flux, _flux(q, br, gamma))
    assert flux[3] == pytest.approx(q[1]*q[3])
    hydro = q.copy()
    hydro[3:] = 0
    sound = np.sqrt(gamma*q[2]/q[0])
    assert _fast_speed(hydro, 0, gamma) == pytest.approx(sound)
    assert _fast_speed(hydro, br, gamma) == pytest.approx(sound)
    assert _fast_speed(q, br, gamma) > _fast_speed(hydro, br, gamma)


def test_radial_flux_constraint_and_force_free_radial_field():
    r = np.linspace(2e10, 4e10, 64)
    solver = MHDSolver(r, br=1e-6)
    out = solver.solve([0, 1000], lambda t: 0, lambda t: 1e-18, lambda t: 1e5,
                       rho_init=np.full(64, 1e-18), v_init=np.zeros(64),
                       T_init=np.full(64, 1e5))
    np.testing.assert_allclose(out['br']*r**2, np.full((2, 64), 1e-6*r[0]**2), rtol=1e-15)
    np.testing.assert_allclose(out['v'], 0, atol=1e-8)
    np.testing.assert_allclose(out['T'], 1e5, rtol=1e-12)


def test_spherical_flux_budgets_and_gravity_work():
    solver = MHDSolver(np.geomspace(2e10, 5e10, 50), br=1e-8)
    r, faces = solver.r, solver.r_int
    q = np.zeros((len(r), 5))
    q[:, 0] = 1e-18*(r[0]/r)**2
    q[:, 1] = 4e5+1e4*np.sin(r/r[0])
    q[:, 2] = 1e-10*(r[0]/r)**3
    q[:, 3], q[:, 4] = 1e-8*r[0]/r, -2e-8*r[0]/r
    U = np.array([_conserved(row, 1.5) for row in q])
    args = (r, faces, solver.magnetic_flux, 1.5, True)
    rhs = _rhs(U, q[0], *args, False)
    f0 = _flux(q[0], solver.magnetic_flux/faces[0]**2, 1.5)
    f1 = _flux(q[-1], solver.magnetic_flux/faces[-1]**2, 1.5)
    for k, power in ((0, 2), (2, 2), (3, 1), (4, 1)):
        weights = np.diff(faces**(power+1))/(power+1)
        expected = faces[0]**power*f0[k]-faces[-1]**power*f1[k]
        np.testing.assert_allclose(np.sum(weights*rhs[:, k]), expected, rtol=1e-12)
    grav_rhs = _rhs(U, q[0], *args, True)
    np.testing.assert_allclose(grav_rhs[:, 1]-rhs[:, 1], -GM_SUN*U[:, 0]/r**2, rtol=1e-12)
    np.testing.assert_allclose(grav_rhs[:, 2]-rhs[:, 2], -GM_SUN*U[:, 1]/r**2, rtol=1e-12)


def test_radial_flow_state_has_no_transverse_velocity_degrees_of_freedom():
    solver = MHDSolver(np.linspace(2e10, 4e10, 32), boundary={'bt': 10e-9})
    out = solver.solve([0, 100], lambda t: 4e5, lambda t: 1e-18, lambda t: 1e5)
    assert solver.U.shape == (32, 5)
    assert set(('br', 'bt', 'bn')).issubset(out)
    assert 'vt' not in out and 'vn' not in out
    with pytest.raises(ValueError, match='boundary keys'):
        MHDSolver(np.linspace(2e10, 4e10, 32), boundary={'vt': 0})


@pytest.mark.parametrize('method', ['mhd-plm-rk2', 'mhd-pcm-euler'])
def test_reduced_perpendicular_shock_tube_is_positive(method):
    n = 240
    x = (np.arange(n)+.5)/n
    left = x < .5
    rho = np.where(left, 1., .125)
    pressure = np.where(left, 1., .1)
    bt = np.where(left, 1., -1.)*np.sqrt(MU0)
    solver = create_solver(1e6+x, gamma=2, method=method,
                           br=.75*np.sqrt(MU0), boundary={'bt': np.sqrt(MU0)})
    out = solver.solve([0, .1], lambda t: 0, lambda t: 1, lambda t: M_P_SI/K_B_SI,
                       v_init=np.zeros(n), rho_init=rho, T_init=pressure/rho*M_P_SI/K_B_SI,
                       mhd_init={'bt': bt, 'bn': np.zeros(n)})
    assert np.all(np.isfinite(out['T'])) and np.min(out['T']) > 0
    assert np.min(out['rho']) > 0
    # Undisturbed regions and the two field polarities survive the solve.
    np.testing.assert_allclose(out['rho'][-1, :20], 1, rtol=1e-4)
    # PCM's diffusive tail reaches farther ahead of the right-going wave.
    np.testing.assert_allclose(out['rho'][-1, -8:], .125, rtol=1e-4)
    assert out['bt'][-1, 20] > 0 and out['bt'][-1, -20] < 0


def test_invalid_inputs_fail_explicitly():
    r = np.linspace(2e10, 4e10, 10)
    for kwargs in ({'br': [1e-8, -1e-8]}, {'gamma': 1}, {'cfl': .9}, {'pui': True}):
        with pytest.raises(ValueError):
            MHDSolver(r, **kwargs)
    with pytest.raises(ValueError, match='strictly increasing'):
        MHDSolver(r).solve([0, 0], lambda t: 4e5, lambda t: 1e-18, lambda t: 1e5)
    with pytest.raises(ValueError, match='scalar'):
        normalize_boundary({'br': np.ones(8)}, 8, 1)
    with pytest.raises(ValueError, match='finite'):
        normalize_boundary({'bt': np.nan}, 8, 1)


def test_perpendicular_fast_shock_rankine_hugoniot_solution():
    from scipy.optimize import brentq
    gamma = 5/3
    upstream = np.array([1., 3., .1, np.sqrt(MU0), 0.])
    def downstream(compression):
        pressure = .1+9*(1-1/compression)+.5*(1-compression**2)
        return np.array([compression, 3/compression, pressure,
                         compression*np.sqrt(MU0), 0.])
    # Solve mass, momentum, induction and energy jumps independently of HLL.
    energy_flux = _flux(upstream, 0., gamma)[2]
    compression = brentq(lambda c: _flux(downstream(c), 0., gamma)[2]-energy_flux, 2., 3.)
    right = downstream(compression)
    np.testing.assert_allclose(_flux(right, 0., gamma), _flux(upstream, 0., gamma), atol=1e-12)
    n = 256
    x = (np.arange(n)+.5)/n
    initial = np.where((x < .5)[:, None], upstream, right)
    solver = MHDSolver(1e6+x, gamma=gamma, boundary={'bt': upstream[3]})
    result = solver.solve([0., .1], lambda t: 3., lambda t: 1., lambda t: .1*M_P_SI/K_B_SI,
                          v_init=initial[:, 1], rho_init=initial[:, 0],
                          T_init=initial[:, 2]/initial[:, 0]*M_P_SI/K_B_SI,
                          mhd_init={'bt': initial[:, 3]})
    # Shock remains at its analytic position with correct asymptotic states.
    assert np.mean(np.abs(result['rho'][-1]-initial[:, 0])) < .02
    # Stay beyond the fast startup transient emitted by shock discretization.
    np.testing.assert_allclose(result['rho'][-1, x > .9], compression, rtol=1e-4)
    np.testing.assert_allclose(result['bt'][-1, x > .9], right[3], rtol=1e-4)


def test_spherical_transverse_field_expansion():
    r = np.linspace(2e10, 4e10, 100)
    solver = MHDSolver(r, boundary={'bt': 1e-10})
    # Negligible gas/magnetic pressure: rho~r^-2, Bt~r^-1, v~constant.
    result = solver.solve([0, 50000], lambda t: 4e5, lambda t: 1e-18, lambda t: 10.)
    interior = slice(5, -5)
    np.testing.assert_allclose(result['bt'][-1, interior],
                               1e-10*r[0]/r[interior], rtol=.015)
    np.testing.assert_allclose(result['rho'][-1, interior],
                               1e-18*(r[0]/r[interior])**2, rtol=.015)


def make_model(**kwargs):
    options = dict(v_boundary=np.full(8, 400.)*u.km/u.s, nlon=8, lon_out=0*u.deg,
                   r_min=30*u.solRad, r_max=45*u.solRad, dr=1*u.solRad,
                   simtime=.06*u.day, solver='mhd', track_cmes=False,
                   mhd_boundary={'br': 10*u.nT, 'bt': -20*u.nT})
    options.update(kwargs)
    return s.SURF(**options)


def test_standard_mhd_plot_has_field_magnitude_fourth_panel():
    import matplotlib.pyplot as plt
    from surf import surf_analysis as sa
    model = make_model()
    model.solve([])
    fig, axes = sa.plot_compressible(
        model, model.time_out[-1], annotateplot=False, bodies=[])
    assert len(axes) == 4
    assert len(fig.axes) == 8  # four polar panels and four colour bars
    assert axes[0].get_position().y0 > axes[2].get_position().y0
    plt.close(fig)


def test_surf_mhd_save_load_and_restart(tmp_path):
    model = make_model()
    model.solve([])
    assert model.compressible and model.bt_grid.shape == model.v_grid.shape
    assert np.all(model.temp_grid.value > 0)
    model._data_dir_ = str(tmp_path)
    path = model.save(tag='mhd')
    loaded, _ = s.load_SURF_run(path)
    for key in ('br', 'bt', 'bn'):
        np.testing.assert_array_equal(getattr(model, key+'_grid'), getattr(loaded, key+'_grid'))
    restart = make_model(simtime=.02*u.day)
    restart.set_initial_state(model.get_final_state())
    restart.solve([])
    np.testing.assert_allclose(restart.bt_grid[0].value, model.bt_grid[-1].value, rtol=1e-12)
    state = model.get_final_state()
    state['mhd']['br'] *= -1
    with pytest.raises(ValueError, match='radial flux'):
        restart.set_initial_state(state)


def test_chunked_mhd_preserves_time_varying_fields():
    boundary = {'br': 10*u.nT, 'time': [-100000, 0, 1500, 2000, 10000]*u.s,
                'bt': [-20, -20, -40, -10, -10]*u.nT}
    continuous, chunked = make_model(mhd_boundary=boundary), make_model(mhd_boundary=boundary)
    continuous.solve([])
    s.solve_chunked(chunked, [], chunk_simtime=.03*u.day, verbose=False)
    np.testing.assert_array_equal(chunked.time_out, continuous.time_out)
    for key in ('br', 'bt', 'bn', 'v', 'rho', 'temp'):
        np.testing.assert_allclose(getattr(chunked, key+'_grid').value,
                                   getattr(continuous, key+'_grid').value, rtol=2e-7, atol=1e-25)


def test_parallel_longitudes_and_cme_tracers():
    kwargs = dict(lon_out=np.nan*u.rad, parallel=True, track_cmes=True,
                  mhd_boundary={'bt': np.linspace(-10, -30, 8)*u.nT})
    model = make_model(**kwargs)
    cme = s.ConeCME(t_launch=.01*u.day, longitude=0*u.deg, latitude=0*u.deg,
                    width=90*u.deg, v=800*u.km/u.s, thickness=1*u.solRad,
                    initial_height=30*u.solRad)
    model.solve([cme])
    assert model.bt_grid.shape[-1] == 8
    assert np.all(np.isfinite(model.bt_grid.value))
    assert np.ptp(model.bt_grid[-1, 0].value) > 1e-9
    assert np.any(np.isfinite(model.cme_particles_r.value))


def test_observer_outputs_physical_magnetic_components(monkeypatch):
    from types import SimpleNamespace
    from surf.surf_analysis import get_observer_timeseries, get_SURF_at_position_HEEQ
    model = make_model()
    model.solve([])
    position = SimpleNamespace(lon=np.zeros(model.nt_out)*u.rad,
                               lon_hae=np.zeros(model.nt_out)*u.rad,
                               r=np.full(model.nt_out, 35.)*u.solRad)
    monkeypatch.setattr(model, 'get_observer', lambda _: position)
    observed = get_observer_timeseries(model, suppress_warning=True)
    mjd = model.time_init.mjd+model.time_out.to_value(u.day)
    generic = get_SURF_at_position_HEEQ(model, mjd, position.r.value, position.lon.value)
    radial_index = np.argmin(abs(model.r.to_value(u.solRad)-35))
    for key in ('br', 'bt', 'bn'):
        unit = u.nT
        expected = getattr(model, key+'_grid')[:, radial_index, 0].to_value(unit)
        np.testing.assert_allclose(observed[key], expected)
        np.testing.assert_allclose(generic[key], expected)
