"""Matched public-interface ConeCME comparison at 1 AU."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from astropy.utils import iers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from surf.surf import SURF, ConeCME
from surf.surf_solvers import M_P_SI

iers.conf.auto_download = False

AMBIENT_SPEED = 410.9823061068116
AMBIENT_DENSITY = 69.38831795273691
SURF_TEMPERATURE = 714993.5756101364/0.6
AMBIENT_BT = 15.615101018053847


def make_cme(speed):
    return ConeCME(
        t_launch=0*u.hour, longitude=0*u.deg, latitude=0*u.deg,
        v=speed*u.km/u.s, width=60*u.deg, initial_height=30*u.solRad,
        cme_fixed_duration=True, fixed_duration=19.6*u.hour,
        profile_type='sinusoidal', label=f'{speed:g} km/s CME')


def make_model(solver):
    nlon = 128
    kwargs = dict(
        v_boundary=np.full(nlon, AMBIENT_SPEED)*u.km/u.s,
        rho_boundary=np.full(nlon, AMBIENT_DENSITY*1e6*M_P_SI)*u.kg/u.m**3,
        temp_boundary=np.full(nlon, SURF_TEMPERATURE)*u.K,
        nlon=nlon, lon_out=0*u.deg, r_min=30*u.solRad,
        r_max=(1.6*u.au).to(u.solRad), dr=1.5*u.solRad,
        simtime=120*u.hour, solver=solver, gamma=1.5,
        track_cmes=True, parallel=False, v_max=3000*u.km/u.s)
    if solver == 'mhd':
        kwargs['mhd_boundary'] = {'br': 0*u.nT, 'bt': AMBIENT_BT*u.nT, 'bn': 0*u.nT}
    return SURF(**kwargs)


def at_au(model, grid, unit):
    radius = model.r.to_value(u.au)
    j = np.searchsorted(radius, 1.)
    w = (1-radius[j-1])/(radius[j]-radius[j-1])
    values = grid.to_value(unit)
    return (1-w)*values[:, j-1, 0]+w*values[:, j, 0]


def run(speed):
    # The exact same user-facing list object is passed to each solve. SURF
    # deep-copies it internally, so one solver cannot mutate another's input.
    cme_list = [make_cme(speed)]
    models, series = {}, {}
    for solver in ('huxt', 'hydro', 'mhd'):
        model = make_model(solver)
        model.solve(cme_list)
        models[solver] = model
        item = {'time_hours': model.time_out.to_value(u.hour),
                'speed_kms': at_au(model, model.v_grid, u.km/u.s)}
        if model.compressible:
            item['density_cm3'] = at_au(model, model.rho_grid, u.kg/u.m**3)/M_P_SI/1e6
            item['temperature_K'] = at_au(model, model.temp_grid, u.K)
        if model.mhd:
            components = [at_au(model, getattr(model, k+'_grid'), u.nT)
                          for k in ('br', 'bt', 'bn')]
            item['B_nT'] = np.sqrt(sum(component**2 for component in components))
        series[solver] = item
    # Verify the standard CME injector produced one identical velocity boundary
    # series for all solver families.
    np.testing.assert_allclose(models['huxt'].input_v_ts, models['hydro'].input_v_ts)
    np.testing.assert_allclose(models['huxt'].input_v_ts, models['mhd'].input_v_ts)
    return series


def save(series, speed, outdir):
    colors = {'huxt': '#4c4c4c', 'hydro': '#d17a22', 'mhd': '#087e8b'}
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    for solver in ('huxt', 'hydro', 'mhd'):
        axes[0].plot(series[solver]['time_hours'], series[solver]['speed_kms'],
                     color=colors[solver], lw=1.8, label=solver.upper() if solver == 'mhd' else solver.title())
    for solver in ('hydro', 'mhd'):
        axes[1].plot(series[solver]['time_hours'], series[solver]['density_cm3'],
                     color=colors[solver], lw=1.8, label=solver.upper() if solver == 'mhd' else solver.title())
        axes[2].plot(series[solver]['time_hours'], series[solver]['temperature_K']/1e6,
                     color=colors[solver], lw=1.8, label=solver.upper() if solver == 'mhd' else solver.title())
    axes[3].plot(series['mhd']['time_hours'], series['mhd']['B_nT'],
                 color=colors['mhd'], lw=1.8, label='MHD')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(frameon=False, loc='upper right'); ax.set_xlim(0, 120)
    axes[0].set_ylabel('Speed\n(km/s)')
    axes[1].set_ylabel('Number density\n(cm$^{-3}$)'); axes[1].set_yscale('log')
    axes[2].set_ylabel('Temperature\n(MK)'); axes[2].set_yscale('log')
    axes[3].set_ylabel('$B$\n(nT)')
    axes[3].set_xlabel(r'Hours after ConeCME launch at 30 $R_\odot$')
    fig.suptitle(f'{speed:g} km/s ConeCME at 1 AU — matched standard SURF interface', fontsize=15)
    fig.text(.5, .945, r'Same vin and ConeCME list • sinusoidal profile • $\Delta r=1.5\,R_\odot$',
             ha='center', fontsize=9, color='#4d5962')
    fig.subplots_adjust(left=.14, right=.97, bottom=.08, top=.91, hspace=.12)
    stem = f'conecme_{speed:g}kms_three_solvers_1au_dr1p5'
    fig.savefig(outdir/f'{stem}.png', dpi=180)
    fig.savefig(outdir/f'{stem}.svg')
    plt.close(fig)
    for solver, item in series.items():
        keys = list(item)
        np.savetxt(outdir/f'conecme_{speed:g}kms_{solver}_1au_dr1p5.csv',
                   np.column_stack([item[k] for k in keys]), delimiter=',',
                   header=','.join(keys), comments='')
    return outdir/f'{stem}.png'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--speed', type=float, default=800.)
    args = parser.parse_args()
    out = Path(__file__).resolve().parents[1]/'outputs'/'sunrunner_fast_cme'
    out.mkdir(parents=True, exist_ok=True)
    print(save(run(args.speed), args.speed, out))
