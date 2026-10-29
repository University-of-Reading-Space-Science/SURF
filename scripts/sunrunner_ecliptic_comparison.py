"""Matched HUXt, hydro, and MHD ecliptic runs for the half-speed CME case.

This extends the sunRunner-like 1-D boundary pulse into longitude as a 60 degree
cone.  The cone width is an explicit modelling assumption: the source 1-D run
does not define an angular width.  At the cone nose, the boundary is exactly the
half-speed case from ``sunrunner_fast_cme.py``.  Away from the nose a smooth
cosine-squared taper is applied.
"""
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.utils import iers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surf.surf import SURF, longitude_grid
from surf.surf_mhd import normalize_boundary
from surf.surf_solvers import M_P_SI
import surf.surf_analysis as sa

iers.conf.auto_download = False

AMBIENT_SPEED = 410.9823061068116
REFERENCE_PEAK_SPEED = AMBIENT_SPEED + 1977.3748818210747
PEAK_SPEED = 0.5 * REFERENCE_PEAK_SPEED
AMBIENT_DENSITY = 69.38831795273691
PEAK_DENSITY = AMBIENT_DENSITY + 1870.90823707962
AMBIENT_BT = 340.5655558402601 * 0.0458505
PEAK_BT = AMBIENT_BT + 2761.226079869457 * 0.0458505
PAPER_TEMPERATURE = 714993.5756101364
SURF_TEMPERATURE = PAPER_TEMPERATURE / 0.6
LAUNCH_HOUR = 12.0
DURATION_HOUR = 19.6
WIDTH_DEG = 60.0
CENTRE_DEG = 0.0
SNAPSHOT_HOUR = 52.0


def boundary_maps(model):
    """Return matched separable sine-squared time/cosine-squared longitude maps."""
    hour = model.model_time.to_value(u.hour)
    phase = (hour - LAUNCH_HOUR) / DURATION_HOUR
    temporal = np.where((phase >= 0) & (phase <= 1), np.sin(np.pi * phase) ** 2, 0.0)
    delta = np.angle(np.exp(1j * (model.lon.to_value(u.rad) - np.deg2rad(CENTRE_DEG))))
    half_width = np.deg2rad(WIDTH_DEG / 2)
    angular = np.where(np.abs(delta) <= half_width,
                       np.cos(0.5 * np.pi * delta / half_width) ** 2, 0.0)
    pulse = temporal[:, None] * angular[None, :]
    speed = AMBIENT_SPEED + (PEAK_SPEED - AMBIENT_SPEED) * pulse
    density = AMBIENT_DENSITY + (PEAK_DENSITY - AMBIENT_DENSITY) * pulse
    bt = AMBIENT_BT + (PEAK_BT - AMBIENT_BT) * pulse
    temperature = np.full_like(speed, SURF_TEMPERATURE)
    return speed, density, temperature, bt, pulse


def configure_boundary(model):
    # Construct SURF's exact spin-up-inclusive time grid before replacing the
    # automatically rotated ambient series with the matched prescribed maps.
    model.ts_from_vlong()
    speed, density, temperature, bt, pulse = boundary_maps(model)
    model.input_v_ts = speed * u.km / u.s
    model.input_v_ts_flag = True
    model.input_iscme_ts = (pulse > 0).astype(int)
    model.input_iscme_ts_flag = True
    if model.compressible:
        model.input_rho_ts = density * 1e6 * M_P_SI * u.kg / u.m**3
        model.input_rho_ts_flag = True
        model.input_temp_ts = temperature * u.K
        model.input_temp_ts_flag = True
    if model.mhd:
        model.mhd_boundary = normalize_boundary(
            {'br': 0 * u.nT, 'time': model.model_time,
             'bt': bt * u.nT, 'bn': 0 * u.nT,
             'vt': 0 * u.km/u.s, 'vn': 0 * u.km/u.s},
            model.nlon_full, model.nlon)


def make_model(solver):
    # Standard SURF full-ecliptic resolution, solving only the requested sector.
    nlon = 128
    model = SURF(
        v_boundary=np.full(nlon, AMBIENT_SPEED) * u.km/u.s,
        rho_boundary=np.full(nlon, AMBIENT_DENSITY * 1e6 * M_P_SI) * u.kg/u.m**3,
        temp_boundary=np.full(nlon, SURF_TEMPERATURE) * u.K,
        r_min=30 * u.solRad, r_max=240 * u.solRad, dr=1.5 * u.solRad,
        simtime=72 * u.hour, nlon=nlon,
        lon_start=315 * u.deg, lon_stop=45 * u.deg,
        solver=solver, gamma=1.5,
        parallel=True, track_cmes=False, v_max=3000 * u.km/u.s,
        mhd_boundary=({'br': 0*u.nT, 'bt': AMBIENT_BT*u.nT}
                      if solver == 'mhd' else None),
        mhd_gravity=False)
    if solver == 'huxt':
        model.alpha = 0.0  # constant-speed ambient, matching the imposed boundary case
    configure_boundary(model)
    return model


def run_model(solver):
    model = make_model(solver)
    started = time.perf_counter()
    model.solve([])
    elapsed = time.perf_counter() - started
    print(f"{solver}: {model.nr} x {model.nlon}, {model.nt_out} outputs, {elapsed:.1f} s",
          flush=True)
    return model, elapsed


def save_standard_plots(models, outdir):
    snapshot = SNAPSHOT_HOUR * u.hour
    fig, _ = sa.plot(models['huxt'], snapshot, minimalplot=False, bodies=[],
                     plotHCS=False, annotateplot=False)
    fig.suptitle(f"SURF-HUXt speed at {SNAPSHOT_HOUR:g} h")
    fig.savefig(outdir / 'huxt_ecliptic_standard.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    for solver in ('hydro', 'mhd'):
        fig, _ = sa.plot_compressible(models[solver], snapshot, minimalplot=False,
                                      bodies=[], annotateplot=False)
        fields = 'speed, density, temperature, $B$' if solver == 'mhd' \
            else 'speed, density, temperature'
        fig.suptitle(f"SURF-{solver.upper()} {fields} at {SNAPSHOT_HOUR:g} h")
        fig.savefig(outdir / f'{solver}_ecliptic_standard.png', dpi=180,
                    bbox_inches='tight')
        plt.close(fig)


def save_speed_comparison(models, outdir):
    """Common colour scale, using the same polar convention as standard SURF plots."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 6), subplot_kw={'projection': 'polar'})
    levels = np.linspace(350, 1200, 35)
    contour = None
    for ax, (name, model) in zip(axes, models.items()):
        it = np.argmin(abs(model.time_out.to_value(u.hour) - SNAPSHOT_HOUR))
        # Put the solved sector back into the standard full grid. Unsolved
        # longitudes remain NaN and consequently blank in the polar plot.
        lon = longitude_grid(nlon=model.nlon_full)[0].to_value(u.rad)
        speed = np.full((model.nr, model.nlon_full), np.nan)
        solved_speed = model.v_grid[it].to_value(u.km/u.s)
        for j, solved_lon in enumerate(model.lon.to_value(u.rad)):
            # Match SURF's standard plotting routine exactly. A circular
            # distance here aliases the final cell onto longitude zero.
            k = np.argmin(np.abs(lon - solved_lon))
            speed[:, k] = solved_speed[:, j]
        theta, radius = np.meshgrid(np.r_[lon, 2*np.pi], model.r.to_value(u.solRad))
        speed = np.c_[speed, speed[:, 0]]
        contour = ax.contourf(theta, radius, speed, levels=levels, cmap='viridis', extend='both')
        ax.set_ylim(0, model.r[-1].to_value(u.solRad))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.plot(0, 0, 'o', color=[1.0, 0.5, 0.25], ms=11)
        ax.set_title('HUXt' if name == 'huxt' else name.upper(), fontsize=14, pad=15)
    cbar = fig.colorbar(contour, ax=axes, orientation='horizontal', fraction=.06,
                        pad=.08, aspect=45)
    cbar.set_label('Radial speed (km/s) — common scale')
    fig.suptitle(f"Matched half-speed CME in the ecliptic plane at {SNAPSHOT_HOUR:g} h",
                 fontsize=16, y=.99)
    fig.text(.5, .925, r"30–240 $R_\odot$; 1.5 $R_\odot$ radial grid; standard 128-cell longitude grid; 315°–45° sector",
             ha='center', color='#48535c')
    fig.subplots_adjust(top=.82, bottom=.19, left=.02, right=.98, wspace=.18)
    fig.savefig(outdir / 'huxt_hydro_mhd_speed_comparison.png', dpi=180,
                bbox_inches='tight')
    fig.savefig(outdir / 'huxt_hydro_mhd_speed_comparison.svg', bbox_inches='tight')
    plt.close(fig)


def main():
    outdir = Path(__file__).resolve().parents[1] / 'outputs' / 'sunrunner_ecliptic_comparison'
    outdir.mkdir(parents=True, exist_ok=True)
    models = {}
    runtimes = {}
    for solver in ('huxt', 'hydro', 'mhd'):
        models[solver], runtimes[solver] = run_model(solver)
    save_standard_plots(models, outdir)
    save_speed_comparison(models, outdir)
    summary = {
        'case': 'half-speed sunRunner-like CME',
        'boundary': {
            'ambient_speed_kms': AMBIENT_SPEED, 'nose_peak_speed_kms': PEAK_SPEED,
            'ambient_density_cm-3': AMBIENT_DENSITY, 'nose_peak_density_cm-3': PEAK_DENSITY,
            'ambient_Bt_nT': AMBIENT_BT, 'nose_peak_Bt_nT': PEAK_BT,
            'paper_temperature_K': PAPER_TEMPERATURE,
            'SURF_internal_temperature_K': SURF_TEMPERATURE,
            'launch_hour': LAUNCH_HOUR, 'duration_hour': DURATION_HOUR,
            'cone_width_deg': WIDTH_DEG, 'cone_centre_deg': CENTRE_DEG,
            'temporal_profile': 'sine-squared', 'angular_profile': 'cosine-squared'},
        'grid': {'r_min_solRad': 30, 'r_max_solRad': 240,
                 'dr_solRad': 1.5, 'nlon_full': 128,
                 'lon_start_deg': 315, 'lon_stop_deg': 45,
                 'nlon_solved': int(models['huxt'].nlon), 'simtime_hour': 72},
        'snapshot_hour': SNAPSHOT_HOUR,
        'runtime_seconds': runtimes,
        'limitations': [
            'SURF ecliptic mode advances independent radial longitude slices; it has no lateral fluxes.',
            'HUXt evolves speed only; hydro also receives density and temperature; MHD additionally receives Bt.',
            'The 60 degree angular extent is an assumption because the source 1-D event has no width.']}
    (outdir / 'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
