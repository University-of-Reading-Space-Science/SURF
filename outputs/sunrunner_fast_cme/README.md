# Fast-CME example with native SURF-MHD

This run uses the extreme fast CME (event 1) in
[Riley & Ben-Nun (2022)](https://doi.org/10.3390/universe8090447).
The time axis starts when the pulse begins at 30 solar radii; it is not the
solar-surface eruption time or an absolute spacecraft observation timestamp.

## Reproduce

From the SURF repository root, with the `surf` environment active:

```powershell
python scripts/sunrunner_fast_cme.py
```

Add `--save-model` to also save the full spatial/time grids as SURF HDF5 files.
The script supplies all four plasma/magnetic boundary profiles explicitly;
it runs the public `SURF(..., solver='mhd')` interface and does not invoke PLUTO.

## Configuration

The published event-1 configuration supplies the unrounded speed, density,
field and temperature values. The pulse adds a sine-squared enhancement to
speed, density and tangential field, with constant boundary temperature.
The density variable is mass density divided by proton mass, consistent with
the reference plotting code.

| Boundary quantity | Ambient | Enhancement |
| --- | ---: | ---: |
| Radial speed (km/s) | 410.9823 | 1977.3749 |
| Density (cm^-3) | 69.3883 | 1870.9082 |
| Tangential field (nT) | 15.6151 | 126.6036 |
| Temperature (K, mu=0.6 convention) | 714993.5756 | 0 |

The main run uses a 0.25-solar-radius grid; the comparison uses 0.5 solar radii.
Both run from 30 solar radii to 1.6 AU, relax for 332 hours, and then evolve
for 120 hours after pulse onset. The observer is linearly interpolated between
the two radial cells bracketing exactly 1 AU. Output cadence is approximately
five minutes. Gamma is 1.5; gravity, radial field and transverse flow are zero.

The 19.6-hour pulse duration and 30-solar-radius inner boundary follow the
paper. The published repository instead specifies about 19.45 hours and
0.13 AU. The outer boundary follows that repository's 1.6-AU setting, keeping
the observer well inside the domain. These choices make this a comparable
simulation, not an exact reproduction of the original calculation.

The source uses `p=rho*k_B*T/(0.6*m_p)`. SURF uses `p=rho*k_B*T_SURF/m_p`,
so the boundary temperature passed to SURF is divided by 0.6 and temperatures
plotted/exported here are multiplied by 0.6. This matches thermal pressure
without changing the SURF equation of state.

The blue curve is the finer solution; the dotted orange curve is the coarser
solution. The dashed grey line is the relaxed ambient value at 1 AU. The wind
is constant in time before the CME, but it accelerates and dilutes with radius.
No observed data or reference-code output is plotted.

## Files

* `sunrunner_fast_cme_1au.png` and `.svg`: four-panel time-series figure.
* `sunrunner_cme_speed_comparison_1au.png` and `.svg`: reference and
  half-peak-speed CMEs overlaid at 0.25 and 1.5 solar-radii resolution.
* `1au_dr0p25.csv`, `1au_dr0p5.csv`: numerical time series with units in headers.
* `1au_dr0p25_speedx0p5.csv`, `1au_dr1p5.csv`, and
  `1au_dr1p5_speedx0p5.csv`: half-speed and 1.5-solar-radius comparison data.
* `run_summary.json`: exact parameters, peak values, numerical arrival marker,
  pre-arrival steady-state check, and resolution differences.

Reference configuration and boundary formula:
[pluto_1.ini](https://github.com/predsci/sunRunner1D-manuscript/blob/main/pluto_1.ini),
[init.c](https://github.com/predsci/sunRunner1D-manuscript/blob/main/src/init.c),
[unit conversions](https://github.com/predsci/sunRunner1D-manuscript/blob/main/util.py).
