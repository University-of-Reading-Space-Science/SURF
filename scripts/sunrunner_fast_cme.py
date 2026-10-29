"""Run sunRunner1D's event-1-like boundary pulse with native SURF-MHD.

From the repository root, in the surf environment:
    python scripts/sunrunner_fast_cme.py

No external simulation code or data download is required. Parameters come
from Riley & Ben-Nun (2022), Table 2, and the published event-1 configuration:
https://doi.org/10.3390/universe8090447
https://github.com/predsci/sunRunner1D-manuscript/blob/main/pluto_1.ini
https://github.com/predsci/sunRunner1D-manuscript/blob/main/src/init.c
https://github.com/predsci/sunRunner1D-manuscript/blob/main/util.py
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.utils import iers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surf.surf import SURF
from surf.surf_solvers import M_P_SI

iers.conf.auto_download = False

PARAMS = {
    'speed_background_kms': 410.9823061068116,
    'speed_enhancement_kms': 1977.3748818210747,
    'density_background_cm3': 69.38831795273691,
    'density_enhancement_cm3': 1870.90823707962,
    'bt_background_nT': 340.5655558402601 * 0.0458505,
    'bt_enhancement_nT': 2761.226079869457 * 0.0458505,
    'temperature_K': 714993.5756101364,
    'mean_molecular_weight': 0.6,
    'gamma': 1.5,
    'pulse_duration_hours': 19.6,
    'spinup_hours': 332.0,
    'inner_radius_solRad': 30.0,
    'outer_radius_AU': 1.6,
    'run_hours': 120.0,
    'input_cadence_seconds': 60.0,
}


def pulse(t_seconds):
    phase = np.asarray(t_seconds) / (PARAMS['pulse_duration_hours'] * 3600)
    return np.where((phase >= 0) & (phase <= 1), np.sin(np.pi*phase)**2, 0.)


def run(dr, outdir, save_model=False, cme_peak_factor=1.0):
    """Use the public SURF class, with prescribed plasma and magnetic series."""
    p = PARAMS
    t = np.arange(-p['spinup_hours']*3600,
                  p['run_hours']*3600 + p['input_cadence_seconds'],
                  p['input_cadence_seconds'])
    f = pulse(t)
    reference_peak = p['speed_background_kms'] + p['speed_enhancement_kms']
    speed_enhancement = reference_peak*cme_peak_factor - p['speed_background_kms']
    if speed_enhancement <= 0:
        raise ValueError('cme_peak_factor must leave the CME faster than the ambient wind')
    speed = p['speed_background_kms'] + speed_enhancement*f
    density = p['density_background_cm3'] + p['density_enhancement_cm3']*f
    bt = p['bt_background_nT'] + p['bt_enhancement_nT']*f
    # Match p=rho k_B T/(mu m_p). SURF internally uses mu=1.
    temperature_surf = p['temperature_K']/p['mean_molecular_weight']
    nlon = 8
    columns = lambda values: np.repeat(np.asarray(values)[:, None], nlon, axis=1)
    dt_scale = max(1, round(300 / ((dr*u.solRad)/(3000*u.km/u.s)).to_value(u.s)))
    model = SURF(
        v_boundary=np.full(nlon, p['speed_background_kms'])*u.km/u.s,
        rho_boundary=np.full(nlon, p['density_background_cm3']*1e6*M_P_SI)*u.kg/u.m**3,
        temp_boundary=np.full(nlon, temperature_surf)*u.K,
        input_t_ts=t*u.s,
        input_v_ts=columns(speed)*u.km/u.s,
        input_rho_ts=columns(density*1e6*M_P_SI)*u.kg/u.m**3,
        input_temp_ts=np.full((len(t), nlon), temperature_surf)*u.K,
        nlon=nlon, lon_out=0*u.deg,
        r_min=p['inner_radius_solRad']*u.solRad, r_max=(p['outer_radius_AU']*u.au).to(u.solRad),
        dr=dr*u.solRad, simtime=p['run_hours']*u.hour, dt_scale=dt_scale,
        solver='mhd', gamma=p['gamma'], track_cmes=False,
        mhd_boundary={'br': 0*u.nT, 'time': t*u.s, 'bt': bt*u.nT},
        mhd_gravity=False,
    )
    print(f'Running dr={dr:g} Rs: {model.nr} shells, {model.nt_out} output times', flush=True)
    started = time.perf_counter()
    model.solve([])  # The CME is the prescribed pulse, not an additional ConeCME.
    print(f'Completed in {time.perf_counter()-started:.1f} s', flush=True)
    r = model.r.to_value(u.au)
    j = np.searchsorted(r, 1.0)
    assert 0 < j < len(r)
    weight = (1-r[j-1])/(r[j]-r[j-1])
    def at_au(grid, unit):
        values = grid.to_value(unit)
        return (1-weight)*values[:, j-1, 0] + weight*values[:, j, 0]
    series = {
        'time_hours': model.time_out.to_value(u.hour),
        'vr_kms': at_au(model.v_grid, u.km/u.s),
        'number_density_cm3': at_au(model.rho_grid, u.kg/u.m**3)/M_P_SI/1e6,
        'Bt_nT': at_au(model.bt_grid, u.nT),
        'temperature_K': at_au(model.temp_grid, u.K)*p['mean_molecular_weight'],
    }
    for key, values in series.items():
        assert np.all(np.isfinite(values)), key
    assert np.all(series['number_density_cm3'] > 0)
    assert np.all(series['temperature_K'] > 0)
    name = f'dr{dr:g}'.replace('.', 'p')
    if cme_peak_factor != 1:
        name += f'_speedx{cme_peak_factor:g}'.replace('.', 'p')
    np.savetxt(outdir / f'1au_{name}.csv', np.column_stack(list(series.values())),
               delimiter=',', header=','.join(series), comments='')
    if save_model:
        model._data_dir_ = str(outdir)
        model.save(tag=f'sunrunner_event1_{name}')
    stats = {'dr_solRad': dr, 'radial_cells': model.nr,
             'cme_peak_factor': cme_peak_factor,
             'boundary_peak_speed_kms': float(reference_peak*cme_peak_factor),
             'output_cadence_seconds': model.dt_out.to_value(u.s),
             'interpolation_radii_AU': [float(r[j-1]), float(r[j])]}
    for key in ('vr_kms', 'number_density_cm3', 'Bt_nT', 'temperature_K'):
        values = series[key]
        stats[key] = {'ambient': float(values[0]), 'peak': float(values.max()),
                      'peak_time_hours': float(series['time_hours'][np.argmax(values)]),
                      'prearrival_relative_drift': float(np.max(abs(values[series['time_hours'] < 10]/values[0]-1)))}
    # Explicit reproducible numerical marker, not the observed event arrival.
    crossing = np.flatnonzero(series['vr_kms'] > series['vr_kms'][0]+50)
    stats['arrival_speed_plus50_hours'] = float(series['time_hours'][crossing[0]]) if crossing.size else None
    return series, stats


def plot_speed_comparison(reference, half_speed, outdir,
                          reference_coarse=None, half_speed_coarse=None):
    """Overlay the reference and half-peak-speed CMEs at 1 AU."""
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fields = [('vr_kms', 'Radial speed\n(km/s)', 1),
              ('number_density_cm3', 'Number density\n(cm$^{-3}$)', 1),
              ('Bt_nT', 'Transverse field\n(nT)', 1),
              ('temperature_K', 'Temperature\n(MK)', 1e6)]
    for ax, (key, label, scale) in zip(axes, fields):
        ax.plot(reference['time_hours'], reference[key]/scale, color='#096c91', lw=1.9,
                label='Reference CME: 2,388 km/s at 30 solar radii')
        ax.plot(half_speed['time_hours'], half_speed[key]/scale, color='#c44e52', lw=1.9,
                label='Half-speed CME: 1,194 km/s at 30 solar radii')
        if reference_coarse is not None:
            ax.plot(reference_coarse['time_hours'], reference_coarse[key]/scale,
                    color='#096c91', lw=1.25, ls=':', alpha=.95,
                    label=r'Reference CME: 1.5 $R_\odot$ grid')
        if half_speed_coarse is not None:
            ax.plot(half_speed_coarse['time_hours'], half_speed_coarse[key]/scale,
                    color='#c44e52', lw=1.25, ls=':', alpha=.95,
                    label=r'Half-speed CME: 1.5 $R_\odot$ grid')
        ax.axhline(reference[key][0]/scale, color='#737d86', lw=1, ls='--',
                   label='Steady ambient wind')
        ax.set_ylabel(label)
        ax.grid(alpha=.17)
        if key in ('number_density_cm3', 'temperature_K'):
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9, loc='upper right')
    axes[-1].set_xlabel('Hours since pulse begins at 30 solar radii')
    axes[-1].set_xlim(0, 120)
    fig.suptitle('CME speed comparison — SURF-MHD at 1 AU', fontsize=16, x=.54, y=.982)
    fig.text(.54, .951, 'Same density, magnetic-field and 19.6 h duration pulses • solid: 0.25 R☉ • dotted: 1.5 R☉',
             ha='center', fontsize=10, color='#48535c')
    fig.text(.14, .018, r'$B_r=0$, $\gamma=1.5$, no gravity; temperature shown with sunRunner $\mu=0.6$.',
             fontsize=9, color='#48535c')
    fig.subplots_adjust(left=.14, right=.97, top=.91, bottom=.09, hspace=.12)
    fig.savefig(outdir/'sunrunner_cme_speed_comparison_1au.png', dpi=180)
    fig.savefig(outdir/'sunrunner_cme_speed_comparison_1au.svg')
    plt.close(fig)


def plot_series(fine, coarse, outdir):
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.facecolor': 'white'})
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    fields = [('vr_kms', 'Radial speed\n(km/s)', 1),
              ('number_density_cm3', 'Number density\n(cm$^{-3}$)', 1),
              ('Bt_nT', 'Transverse field\n(nT)', 1),
              ('temperature_K', 'Temperature\n(MK)', 1e6)]
    for ax, (key, label, scale) in zip(axes, fields):
        ax.plot(fine['time_hours'], fine[key]/scale, color='#096c91', lw=1.9,
                label='SURF-MHD (0.25 solar radii)')
        ax.plot(coarse['time_hours'], coarse[key]/scale, color='#cb852b',
                lw=1.1, ls=':', alpha=.8, label='Resolution check (0.5 solar radii)')
        ax.axhline(fine[key][0]/scale, color='#737d86', lw=1, ls='--', label='Steady ambient wind')
        ax.set_ylabel(label)
        ax.grid(alpha=.17)
        if key in ('number_density_cm3', 'temperature_K'):
            ax.set_yscale('log')
        else:
            ax.set_ylim(bottom=0)
    axes[0].legend(frameon=False, fontsize=9, loc='upper right')
    axes[-1].set_xlabel('Hours since pulse begins at 30 solar radii')
    axes[-1].set_xlim(0, 120)
    fig.suptitle('Fast CME into steady solar wind — SURF-MHD at 1 AU', fontsize=16, x=.54, y=.982)
    fig.text(.54, .951, 'sunRunner1D event 1 parameters • 19.6 h sine-squared boundary pulse',
             ha='center', fontsize=10, color='#48535c')
    fig.text(.14, .018, r'$B_r=0$, $\gamma=1.5$, no gravity; temperature shown with sunRunner $\mu=0.6$. '
             '\nNative SURF calculation; comparable setup, not a reproduction of PLUTO output.',
             fontsize=9, color='#48535c')
    fig.subplots_adjust(left=.14, right=.97, top=.91, bottom=.10, hspace=.12)
    fig.savefig(outdir/'sunrunner_fast_cme_1au.png', dpi=180)
    fig.savefig(outdir/'sunrunner_fast_cme_1au.svg')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'outputs'/'sunrunner_fast_cme')
    parser.add_argument('--save-model', action='store_true', help='Also save full SURF HDF5 grids')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    coarse, coarse_stats = run(.5, args.output, args.save_model)
    fine, fine_stats = run(.25, args.output, args.save_model)
    half_speed, half_speed_stats = run(.25, args.output, args.save_model, cme_peak_factor=.5)
    standard, standard_stats = run(1.5, args.output, args.save_model)
    standard_half, standard_half_stats = run(1.5, args.output, args.save_model,
                                             cme_peak_factor=.5)
    comparison = {}
    for key in ('vr_kms', 'number_density_cm3', 'Bt_nT', 'temperature_K'):
        interpolated = np.interp(fine['time_hours'], coarse['time_hours'], coarse[key])
        comparison[key] = {'relative_peak_difference': float(abs(coarse[key].max()/fine[key].max()-1)),
                           'mean_abs_difference_normalized_by_peak': float(np.mean(abs(interpolated-fine[key]))/fine[key].max())}
    metadata = {'parameters': PARAMS, 'fine_run': fine_stats, 'coarse_run': coarse_stats,
                'half_speed_run': half_speed_stats,
                'standard_resolution_run': standard_stats,
                'standard_resolution_half_speed_run': standard_half_stats,
                'resolution_comparison': comparison,
                'source_paper': 'https://doi.org/10.3390/universe8090447',
                'source_configuration': 'https://github.com/predsci/sunRunner1D-manuscript/blob/main/pluto_1.ini',
                'notes': ['30 Rs inner boundary and 19.6 h duration follow the paper; repository configuration uses 0.13 AU and about 19.45 h.',
                          'Input density is rho/mp, as in the reference plotting script.',
                          'SURF internal T=T_paper/0.6 matches reference thermal pressure; plotted T=0.6*T_SURF.',
                          'HLL/PLM/RK2 differs from the reference Roe/LimO3/RK3 scheme.',
                          'Radial interpolation samples exactly 1 AU; no observed data or PLUTO outputs are plotted.']}
    (args.output/'run_summary.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    plot_series(fine, coarse, args.output)
    plot_speed_comparison(fine, half_speed, args.output, standard, standard_half)
    print(json.dumps(metadata, indent=2), flush=True)
    print(f'Plot saved to {args.output / "sunrunner_fast_cme_1au.png"}', flush=True)


if __name__ == '__main__':
    main()
