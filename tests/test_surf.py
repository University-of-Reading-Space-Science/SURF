"""Regression and verification tests for the SURF solvers.

Two kinds of test live here:

* Verification tests check behaviour that any correct version of SURF must
  reproduce, such as the analytic HUXt steady state, mass conservation in the
  hydro solver, and agreement between serial, parallel and saved/loaded runs.
* Regression tests rerun a set of reference simulations covering the HUXt,
  hydro and MHD solvers and compare them with
  ``reference_data/surf_regression.npz``. They flag any change in SURF's
  numerical output, intended or not.

If a change to the output is intended (for example a bug fix), regenerate the
reference data with ``python tests/generate_reference_data.py`` and record the
reason in the commit message.
"""
import datetime
import pathlib

import numpy as np
import pytest
from astropy import units as u
from astropy.utils import iers

import surf.surf as s
import surf.surf_inputs as sin

REFERENCE_FILE = pathlib.Path(__file__).parent / "reference_data" / "surf_regression.npz"

# Relative tolerance for regression comparisons. Each field's absolute
# tolerance is RTOL times its largest magnitude, which allows for
# floating-point differences between platforms but not for real changes.
RTOL = 1e-6

# Regression outputs are subsampled in time and longitude to keep the
# reference file small. All radial points are kept.
N_TIME_SAMPLES = 12
N_LON_SAMPLES = 16

NLON = 128
_LONS = np.linspace(0, 2 * np.pi, NLON, endpoint=False)
V_BOUNDARY = (400 + 150 * np.sin(_LONS) + 60 * np.cos(3 * _LONS)) * (u.km / u.s)
B_BOUNDARY = np.sign(np.cos(_LONS + 0.4))
STREAK_CARR = np.deg2rad([30.0, 150.0, 270.0]) * u.rad
MHD_BOUNDARY = {"br": 10 * u.nT, "bt": -20 * u.nT, "bn": 5 * u.nT}


@pytest.fixture(autouse=True)
def no_iers_download():
    """Keep the tests offline."""
    with iers.conf.set_temp("auto_download", False):
        yield


def make_cmes():
    """Return two CMEs that between them use every ConeCME option."""
    return [
        s.ConeCME(t_launch=0.4 * u.day, longitude=0 * u.deg, latitude=0 * u.deg,
                  width=60 * u.deg, v=1000 * (u.km / u.s), thickness=5 * u.solRad),
        s.ConeCME(t_launch=1.2 * u.day, longitude=-40 * u.deg, latitude=5 * u.deg,
                  width=45 * u.deg, v=700 * (u.km / u.s), thickness=2 * u.solRad,
                  cme_expansion=True, cme_fixed_duration=False,
                  profile_type="sinusoidal"),
    ]


def _model(**kwargs):
    """Return a SURF model with the shared test defaults."""
    settings = dict(v_boundary=V_BOUNDARY, cr_num=2080, cr_lon_init=180 * u.deg,
                    simtime=3 * u.day, dt_scale=4)
    settings.update(kwargs)
    return s.SURF(**settings)


# ----------------------------------------------------------------------------
# Reference simulations
# ----------------------------------------------------------------------------

def case_huxt_2d():
    model = _model(b_boundary=B_BOUNDARY)
    model.solve(make_cmes(), streak_carr=STREAK_CARR)
    return model


def case_huxt_sidereal_wedge():
    model = _model(b_boundary=B_BOUNDARY, frame="sidereal",
                   lon_start=270 * u.deg, lon_stop=90 * u.deg)
    model.solve(make_cmes())
    return model


def case_huxt_pui_1d():
    model = _model(solver="huxt-pui", lon_out=0 * u.deg, r_max=1500 * u.solRad)
    model.solve(make_cmes())
    return model


def case_huxt_time_dependent():
    n_days = 8
    time_grid = 59610.0 + np.arange(n_days)  # MJD, starting 2022-02-01
    drift = np.linspace(0, np.pi / 2, n_days)
    vgrid = 400 + 150 * np.sin(_LONS[:, None] + drift[None, :])
    bgrid = np.sign(np.cos(_LONS[:, None] + 0.4 + drift[None, :]))
    model = sin.set_time_dependent_boundary(
        vgrid * (u.km / u.s), time_grid, datetime.datetime(2022, 2, 3),
        3 * u.day, r_min=21.5 * u.solRad, r_max=240 * u.solRad, dt_scale=4,
        frame="sidereal", bgrid_Carr=bgrid)
    model.solve(make_cmes())
    return model


def case_huxt_chunked():
    model = _model(b_boundary=B_BOUNDARY, lon_start=300 * u.deg, lon_stop=60 * u.deg)
    s.solve_chunked(model, make_cmes(), chunk_simtime=1 * u.day,
                    streak_carr=STREAK_CARR, verbose=False)
    return model


def case_hydro_wedge():
    model = _model(solver="hydro", b_boundary=B_BOUNDARY,
                   lon_start=350 * u.deg, lon_stop=30 * u.deg)
    model.solve(make_cmes(), streak_carr=STREAK_CARR)
    return model


def case_hydro_pcm_1d():
    model = _model(solver="hydro-pcm", lon_out=0 * u.deg)
    model.solve(make_cmes())
    return model


def case_hydro_pui_1d():
    model = _model(solver="hydro-pui", lon_out=0 * u.deg, r_max=600 * u.solRad)
    model.solve(make_cmes())
    return model


def case_hydro_chunked_1d():
    model = _model(solver="hydro", lon_out=0 * u.deg)
    s.solve_chunked(model, make_cmes(), chunk_simtime=1 * u.day, verbose=False)
    return model


def case_mhd_1d():
    model = _model(solver="mhd", lon_out=0 * u.deg, mhd_boundary=MHD_BOUNDARY)
    model.solve(make_cmes())
    return model


CASES = {
    "huxt_2d": case_huxt_2d,
    "huxt_sidereal_wedge": case_huxt_sidereal_wedge,
    "huxt_pui_1d": case_huxt_pui_1d,
    "huxt_time_dependent": case_huxt_time_dependent,
    "huxt_chunked": case_huxt_chunked,
    "hydro_wedge": case_hydro_wedge,
    "hydro_pcm_1d": case_hydro_pcm_1d,
    "hydro_pui_1d": case_hydro_pui_1d,
    "hydro_chunked_1d": case_hydro_chunked_1d,
    "mhd_1d": case_mhd_1d,
}


def _values(array, unit=None):
    """Return a plain float array, converting Quantities to ``unit`` if given."""
    if isinstance(array, u.Quantity):
        array = array.to_value(unit) if unit is not None else array.value
    return np.asarray(array, dtype=float)


def summarise(model):
    """Reduce a solved model to the arrays stored in the reference file."""
    it = np.unique(np.linspace(0, model.nt_out - 1, N_TIME_SAMPLES).round().astype(int))
    il = np.unique(np.linspace(0, model.nlon - 1, min(model.nlon, N_LON_SAMPLES))
                   .round().astype(int))

    def grid(array, unit=None):  # (time, r, lon)
        return _values(array, unit)[it][:, :, il]

    def particles(array):  # (n, time, 2, lon)
        return _values(array)[:, it][..., il]

    out = {
        "time_out_s": _values(model.time_out, u.s),
        "lon_rad": np.atleast_1d(_values(model.lon, u.rad)),
        "v_kms": grid(model.v_grid, u.km / u.s),
        "cme_r_km": particles(model.cme_particles_r),
        "cme_v_kms": particles(model.cme_particles_v),
    }
    if model.compressible:
        out["rho_kgm3"] = grid(model.rho_grid, u.kg / u.m ** 3)
        out["temp_K"] = grid(model.temp_grid, u.K)
    if model.mhd:
        for key in ("br", "bt", "bn"):
            out[key + "_T"] = grid(getattr(model, key + "_grid"), u.T)
    if model.track_b:
        out["b_polarity"] = grid(model.b_grid)
        out["hcs_r_km"] = particles(model.hcs_particles_r)
    if model.track_streak:
        out["streak_r_km"] = _values(model.streak_particles_r)[it][..., il]

    arrivals = []
    for cme in model.cmes:
        stats = cme.compute_arrival_at_body("EARTH")
        arrivals.append([
            float(stats["hit"]),
            stats["t_transit"].to_value(u.day),
            stats["v"].to_value(u.km / u.s),
            stats["r"].to_value(u.solRad),
        ])
    out["cme_arrival"] = np.array(arrivals, dtype=float).reshape(-1, 4)
    return out


# ----------------------------------------------------------------------------
# Regression tests
# ----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reference():
    if not REFERENCE_FILE.exists():
        pytest.fail(f"Missing {REFERENCE_FILE}; run tests/generate_reference_data.py")
    with np.load(REFERENCE_FILE) as data:
        return {key: data[key] for key in data.files}


def _assert_matches(actual, expected, name):
    assert actual.shape == expected.shape, f"{name}: shape {actual.shape} != {expected.shape}"
    nan_actual, nan_expected = np.isnan(actual), np.isnan(expected)
    assert np.array_equal(nan_actual, nan_expected), (
        f"{name}: NaN pattern differs at {np.count_nonzero(nan_actual != nan_expected)} points")
    finite = expected[~nan_expected]
    scale = np.max(np.abs(finite)) if finite.size else 0.0
    np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=RTOL * scale,
                               equal_nan=True, err_msg=name)


@pytest.mark.parametrize("case", CASES)
def test_regression(case, reference):
    actual = summarise(CASES[case]())
    prefix = case + "/"
    expected = {key[len(prefix):]: value for key, value in reference.items()
                if key.startswith(prefix)}
    assert expected, f"No reference data for {case}; run tests/generate_reference_data.py"
    assert set(actual) == set(expected), "Different outputs from the reference run"
    for key in sorted(expected):
        _assert_matches(actual[key], expected[key], f"{case}/{key}")


# ----------------------------------------------------------------------------
# Verification tests
# ----------------------------------------------------------------------------

def test_huxt_steady_state_matches_analytic_acceleration():
    """A uniform boundary relaxes to HUXt's analytic residual-acceleration profile."""
    model = _model(v_boundary=np.full(NLON, 400.0) * (u.km / u.s),
                   lon_out=0 * u.deg, simtime=1 * u.day)
    model.solve([])

    constants = s.surf_constants()
    alpha = constants["alpha"].value
    r_accel = constants["r_accel"].to_value(u.solRad)
    profile = 1 + alpha * (1 - np.exp(-model.rrel.to_value(u.solRad) / r_accel))
    v_analytic = 400.0 * profile / profile[0]

    v_model = model.v_grid[-1, :, 0].to_value(u.km / u.s)
    np.testing.assert_allclose(v_model, v_analytic, rtol=1e-3)


def test_hydro_steady_state_conserves_mass_flux():
    """In a steady, uniform hydro wind rho * v * r**2 is constant with radius."""
    model = _model(v_boundary=np.full(NLON, 400.0) * (u.km / u.s),
                   lon_out=0 * u.deg, simtime=2 * u.day, solver="hydro")
    model.solve([])

    flux = (model.rho_grid[-1, :, 0] * model.v_grid[-1, :, 0] * model.r ** 2).value
    assert np.ptp(flux) / np.mean(flux) < 0.02
    assert np.all(np.diff(model.v_grid[-1, :, 0].value) >= 0)


def test_parallel_solve_matches_serial():
    def run(parallel):
        model = _model(b_boundary=B_BOUNDARY, lon_start=300 * u.deg, lon_stop=60 * u.deg,
                       parallel=parallel)
        model.solve(make_cmes(), streak_carr=STREAK_CARR)
        return model

    serial, parallel = run(False), run(True)
    np.testing.assert_array_equal(parallel.v_grid.value, serial.v_grid.value)
    np.testing.assert_array_equal(parallel.b_grid, serial.b_grid)
    np.testing.assert_array_equal(parallel.cme_particles_r.value, serial.cme_particles_r.value)
    np.testing.assert_array_equal(parallel.streak_particles_r.value,
                                  serial.streak_particles_r.value)


@pytest.mark.parametrize("solver", ["huxt", "hydro"])
def test_save_and_load_round_trip(solver, tmp_path):
    if solver == "huxt":
        model = _model(b_boundary=B_BOUNDARY, lon_start=330 * u.deg, lon_stop=30 * u.deg)
        model.solve(make_cmes(), streak_carr=STREAK_CARR)
    else:
        model = _model(solver="hydro", lon_out=0 * u.deg)
        model.solve(make_cmes())
    model._data_dir_ = str(tmp_path)  # keep the test output out of the user data folder
    loaded, cmes = s.load_SURF_run(model.save(tag="roundtrip"))

    assert loaded.solver == model.solver
    np.testing.assert_array_equal(loaded.time_out.value, model.time_out.value)
    np.testing.assert_array_equal(loaded.lon.value, model.lon.value)
    np.testing.assert_array_equal(loaded.v_grid.value, model.v_grid.value)
    np.testing.assert_array_equal(_values(loaded.cme_particles_r),
                                  _values(model.cme_particles_r))
    if model.compressible:
        np.testing.assert_array_equal(loaded.rho_grid.value, model.rho_grid.value)
        np.testing.assert_array_equal(loaded.temp_grid.value, model.temp_grid.value)
    if model.track_b:
        np.testing.assert_array_equal(loaded.b_grid, model.b_grid)
        np.testing.assert_array_equal(_values(loaded.hcs_particles_r),
                                      _values(model.hcs_particles_r))
    if model.track_streak:
        np.testing.assert_array_equal(_values(loaded.streak_particles_r),
                                      _values(model.streak_particles_r))

    assert len(cmes) == len(model.cmes)
    for loaded_cme, cme in zip(cmes, model.cmes, strict=True):
        assert loaded_cme.v == cme.v
        assert loaded_cme.longitude == cme.longitude
        for j, coords in cme.coords.items():
            np.testing.assert_array_equal(loaded_cme.coords[j]["r"].to_value(u.km),
                                          coords["r"].to_value(u.km))


@pytest.mark.xfail(strict=True, reason=(
    "Known issue: solve_chunked() is offset by one model step at chunk joins, so "
    "it does not reproduce a single solve() as its docstring claims. Remove this "
    "marker once that is fixed."))
def test_chunked_solve_matches_single_solve():
    settings = dict(b_boundary=B_BOUNDARY, lon_start=300 * u.deg, lon_stop=60 * u.deg)
    single = _model(**settings)
    single.solve(make_cmes())
    chunked = _model(**settings)
    s.solve_chunked(chunked, make_cmes(), chunk_simtime=1 * u.day, verbose=False)

    np.testing.assert_array_equal(chunked.time_out.value, single.time_out.value)
    np.testing.assert_allclose(chunked.v_grid.value, single.v_grid.value, rtol=1e-9)
