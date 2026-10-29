"""Run the half-speed CME into an example WSA variable ambient wind."""
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.utils import iers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surf.surf import SURF, ConeCME, longitude_grid
from surf import surf_analysis as sa
from surf import surf_inputs as si

iers.conf.auto_download = False

ROOT = Path(__file__).resolve().parents[1]
WSA_FILE = ROOT / 'surf' / 'data' / 'example_inputs' / 'wsa_gong_2024050906.fits'
OUTDIR = ROOT / 'outputs' / 'wsa_variable_cme_comparison'
PEAK_SPEED = 1194.178593963943
AMBIENT_BT_NT = 15.615101018053847
LAUNCH_HOUR, DURATION_HOUR = 0.0, 19.6
WIDTH_DEG, SNAPSHOT_HOUR = 60.0, 52.0


def wsa_boundary():
    source = si.get_WSA_long_profile(WSA_FILE, lat=0*u.deg).to_value(u.km/u.s)
    source_lon = (np.arange(len(source)) + .5) * 2*np.pi/len(source)
    target_lon = longitude_grid(nlon=128)[0].to_value(u.rad)
    speed = np.interp(target_lon, source_lon, source, period=2*np.pi)
    polarity_source = si.get_WSA_br_long_profile(WSA_FILE, lat=0*u.deg)
    polarity = np.sign(np.interp(target_lon, source_lon, polarity_source, period=2*np.pi))
    polarity[polarity == 0] = 1
    return speed, polarity


def make_cme():
    return ConeCME(
        t_launch=LAUNCH_HOUR*u.hour, longitude=0*u.deg, latitude=0*u.deg,
        v=PEAK_SPEED*u.km/u.s, width=WIDTH_DEG*u.deg,
        initial_height=30*u.solRad, cme_fixed_duration=True,
        fixed_duration=DURATION_HOUR*u.hour, profile_type='sinusoidal',
        label='half-speed CME')


def make_model(solver, speed, polarity):
    model = SURF(
        v_boundary=speed*u.km/u.s, b_boundary=polarity,
        cr_num=2284, nlon=128, lon_start=315*u.deg, lon_stop=45*u.deg,
        r_min=30*u.solRad, r_max=240*u.solRad, dr=1.5*u.solRad,
        simtime=72*u.hour, solver=solver, gamma=1.5, parallel=True,
        track_cmes=True, v_max=3000*u.km/u.s,
        mhd_boundary=({'br': 0*u.nT, 'bt': AMBIENT_BT_NT*u.nT}
                      if solver == 'mhd' else None))
    return model


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    speed, polarity = wsa_boundary()
    models, runtimes = {}, {}
    cme_list = [make_cme()]
    for solver in ('huxt', 'hydro', 'mhd'):
        models[solver] = make_model(solver, speed, polarity)
        start = time.perf_counter()
        models[solver].solve(cme_list)
        runtimes[solver] = time.perf_counter()-start
        print(f'{solver}: {runtimes[solver]:.1f} s', flush=True)
    np.testing.assert_allclose(models['huxt'].input_v_ts, models['hydro'].input_v_ts)
    np.testing.assert_allclose(models['huxt'].input_v_ts, models['mhd'].input_v_ts)
    snapshot = SNAPSHOT_HOUR*u.hour
    fig, _ = sa.plot(models['huxt'], snapshot, bodies=[], annotateplot=False, plotHCS=False)
    fig.suptitle(f'WSA variable wind + half-speed CME: HUXt at {SNAPSHOT_HOUR:g} h')
    fig.savefig(OUTDIR/'wsa_huxt_standard.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    for solver in ('hydro', 'mhd'):
        fig, _ = sa.plot_compressible(models[solver], snapshot, bodies=[], annotateplot=False)
        fields = 'speed, density, temperature, $B$' if solver == 'mhd' else 'speed, density, temperature'
        fig.suptitle(f'WSA variable wind + half-speed CME: {solver.upper()} {fields} at {SNAPSHOT_HOUR:g} h')
        fig.savefig(OUTDIR/f'wsa_{solver}_standard.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
    summary = {
        'wsa_file': str(WSA_FILE.relative_to(ROOT)), 'carrington_rotation': 2284,
        'wsa_equatorial_speed_kms': {'min': float(speed.min()), 'max': float(speed.max()),
                                     'median': float(np.median(speed))},
        'grid': {'nlon_full': 128, 'lon_start_deg': 315, 'lon_stop_deg': 45,
                 'nlon_solved': int(models['mhd'].nlon), 'dr_solRad': 1.5},
        'cme': {'peak_speed_kms': PEAK_SPEED, 'duration_hour': DURATION_HOUR,
                'width_deg': WIDTH_DEG, 'launch_hour': LAUNCH_HOUR,
                'density_cm-3': 600.0, 'temperature_K': 1e6,
                'profile_type': 'sinusoidal', 'interface': 'SURF.solve([ConeCME])'},
        'runtime_seconds': runtimes,
        'magnetic_boundary_note': ('WSA Br is used only as SURF polarity metadata. Physical MHD '
                                   'uses the explicit sunRunner Bt boundary; Br=Bn=0.')}
    (OUTDIR/'run_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
