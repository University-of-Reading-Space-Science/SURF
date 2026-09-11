"""Tests for pick-up ion deceleration in the HUXt and hydro solvers."""

import numpy as np

from surf.surf import _upwind_step_, _upwind_step_pui_, surf_constants
from surf.surf_solvers import (
    M_P_SI,
    PUI_REFERENCE_RADIUS_M,
    PUI_SLOWDOWN_PER_AU,
    _apply_pui_source,
    _prim_to_cons,
)
from surf.surf_insitu import _is_compressible_solver


def test_pui_solver_names_are_public():
    solvers = surf_constants()["valid_solvers"]
    assert "huxt-pui" in solvers
    assert "hydro-pui" in solvers
    assert "hydro-pcm-pui" in solvers
    assert not _is_compressible_solver("huxt-pui")
    assert _is_compressible_solver("hydro-pui")
    assert _is_compressible_solver("hydro-pcm-pui")


def test_huxt_pui_decelerates_continuously_beyond_one_au():
    au_km = 1.496e8
    rgrid = np.array([0.5, 1.0, 1.5, 2.0]) * au_km
    rrel = np.arange(4, dtype=float)
    v_up = np.full(3, 400.0)
    v_dn = np.full(3, 400.0)
    # A half-cell Courant step must accumulate only half a cell's slowdown.
    dtdr = 0.5 / 400.0
    baseline = _upwind_step_(v_up, v_dn, dtdr, 0.0, 1.0, rrel)
    pui = _upwind_step_pui_(v_up, v_dn, dtdr, 0.0, 1.0, rrel, rgrid)

    np.testing.assert_allclose(pui[0], baseline[0])
    assert pui[1] < baseline[1]
    assert pui[2] < baseline[2]
    expected_factor = (1.0 - 0.0027 * 1.25) / (1.0 - 0.0027 * 1.0)
    np.testing.assert_allclose(pui[2] / baseline[2], expected_factor)


def test_huxt_pui_source_is_zero_when_no_time_elapses():
    au_km = 1.496e8
    rgrid = np.array([1.0, 1.5, 2.0]) * au_km
    rrel = np.arange(3, dtype=float)
    velocity = np.full(2, 400.0)

    baseline = _upwind_step_(velocity, velocity, 0.0, 0.0, 1.0, rrel)
    pui = _upwind_step_pui_(
        velocity, velocity, 0.0, 0.0, 1.0, rrel, rgrid)

    np.testing.assert_allclose(pui, baseline)


def test_pui_radial_trend_matches_elliott_2026():
    """The mean trend is 0.27%/AU and about 13% by 58 AU."""
    slowdown = lambda radius_au: PUI_SLOWDOWN_PER_AU * (radius_au - 1.0)
    np.testing.assert_allclose(slowdown(30.0), 0.0783)
    np.testing.assert_allclose(slowdown(50.0), 0.1323)
    np.testing.assert_allclose(slowdown(58.0), 0.1539)
    assert 0.13 <= slowdown(50.0) <= 0.15
    assert 0.13 <= slowdown(58.0) <= 0.16


def test_hydro_pui_reduces_bulk_speed_but_not_internal_energy():
    rho = 1e-23
    velocity = 400e3
    pressure = rho * 1.380649e-23 * 1e5 / M_P_SI
    state = np.vstack([
        _prim_to_cons(np.array([rho, velocity, pressure]), 1.5),
        _prim_to_cons(np.array([rho, velocity, pressure]), 1.5),
    ])
    radii = np.array([PUI_REFERENCE_RADIUS_M - 1.0, PUI_REFERENCE_RADIUS_M + 1.0])
    before = state.copy()
    after = _apply_pui_source(state.copy(), radii, 2, 3600.0, True)

    np.testing.assert_allclose(after[0], before[0])
    assert after[1, 1] / after[1, 0] < velocity
    internal_before = before[1, 2] - 0.5 * before[1, 1] ** 2 / before[1, 0]
    internal_after = after[1, 2] - 0.5 * after[1, 1] ** 2 / after[1, 0]
    np.testing.assert_allclose(internal_after, internal_before)


def test_disabled_hydro_pui_source_is_noop():
    state = np.array([[1.0, 2.0, 5.0]])
    result = _apply_pui_source(
        state.copy(), np.array([PUI_REFERENCE_RADIUS_M + 1.0]), 1, 3600.0, False
    )
    np.testing.assert_array_equal(result, state)
