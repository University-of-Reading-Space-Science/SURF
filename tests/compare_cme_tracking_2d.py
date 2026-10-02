"""Compare one CME run using HUXt and hydro, with and without CME tracking."""

import time

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

import surf.surf as s
import surf.surf_analysis as sa


# The same ambient wind and CME are used in all six runs.
nlon = 128
v_boundary = np.ones(nlon) * 400 * u.km / u.s
b_boundary = np.ones(nlon)
b_boundary[nlon // 2:] = -1
streak_carr = np.arange(0, 360, 10) * u.deg


def make_cme():
    return s.ConeCME(
        t_launch=0.5 * u.day,
        longitude=0 * u.deg,
        latitude=0 * u.deg,
        width=60 * u.deg,
        v=1000 * u.km / u.s,
        thickness=5 * u.solRad,
    )


def solve_once(solver, track_cmes, other_particles):
    model_kwargs = dict(
        v_boundary=v_boundary,
        simtime=2.5 * u.day,
        dt_scale=4,
        nlon=nlon,
        solver=solver,
        track_cmes=track_cmes,
    )
    if other_particles:
        model_kwargs["b_boundary"] = b_boundary
    model = s.SURF(**model_kwargs)

    s.solve_chunked(
        model,
        [make_cme()],
        chunk_simtime=1 * u.day,
        streak_carr=streak_carr if other_particles else np.array([]) * u.deg,
        verbose=False,
    )
    return model


def run(solver, track_cmes, other_particles):
    # First run is deliberately discarded so compilation and one-time setup
    # are not included in the reported timing.
    print(
        f"Warming up {solver}, track_cmes={track_cmes}, "
        f"HCS/streaklines={other_particles}",
        flush=True,
    )
    solve_once(solver, track_cmes, other_particles)

    start = time.perf_counter()
    model = solve_once(solver, track_cmes, other_particles)
    elapsed = time.perf_counter() - start
    print(
        f"{solver}, track_cmes={track_cmes}, "
        f"HCS/streaklines={other_particles}: {elapsed:.2f} s"
    )
    return model, elapsed


huxt_no_particles, t1 = run("huxt", False, False)
hydro_no_particles, t2 = run("hydro", False, False)

huxt_hcs_streak, t3 = run("huxt", False, True)
huxt_all_particles, t4 = run("huxt", True, True)
hydro_hcs_streak, t5 = run("hydro", False, True)
hydro_all_particles, t6 = run("hydro", True, True)


# Plot the final velocity solution, HCS, and streaklines.
cases = [
    (huxt_no_particles, "HUXt: no particles", t1),
    (huxt_hcs_streak, "HUXt: HCS + streaklines", t3),
    (huxt_all_particles, "HUXt: HCS + streaklines + CME", t4),
    (hydro_no_particles, "hydro: no particles", t2),
    (hydro_hcs_streak, "hydro: HCS + streaklines", t5),
    (hydro_all_particles, "hydro: HCS + streaklines + CME", t6),
]

print("\nTiming summary")
print("-" * 62)
for _, label, elapsed in cases:
    print(f"{label:<45} {elapsed:>10.2f} s")
print("-" * 62)

fig, axes = plt.subplots(
    2, 3, figsize=(17, 11), subplot_kw={"projection": "polar"}
)
for ax, (model, title, elapsed) in zip(axes.flat, cases):
    sa.plot(
        model,
        model.time_out[-1],
        fighandle=fig,
        axhandle=ax,
        minimalplot=False,
        annotateplot=False,
        plotHCS=True,
        bodies=[],
    )
    ax.set_title(f"{title}\nsolve time = {elapsed:.2f} s")

fig.tight_layout()
#fig.savefig("cme_tracking_comparison.png", dpi=150)
plt.show()
