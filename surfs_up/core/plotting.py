"""Plotting helpers used by the SURFs_UP web interface."""

from __future__ import annotations

import astropy.units as u
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


SPEED_LIMITS_KMS = (300, 900)
TIMESERIES_FIGSIZE = (10, 6.25)
TIMESERIES_LEGEND_FONTSIZE = 12


def full_run_name(simulation, *, ambient: bool = False):
    """Build the descriptive run name used by maps and time-series legends."""
    source = str(simulation.ambient.get("source", "Ideal"))
    ambient_names = {
        "user_specified": "Ideal",
        "mas": "MAS",
        "wsa": "WSA",
        "wsa_iswa": "WSA",
        "cortom": "CorTom",
        "omni": "OMNI",
        "insitu_backmapped": str(
            simulation.ambient.get("spacecraft", "OMNI")
        ).replace("STEREO-A", "STA"),
    }
    parts = [ambient_names.get(source, source)]
    if not ambient:
        if simulation.model.get("grab_donki_at_run_start") or any(
            str(cme.get("source", "")).lower() == "donki"
            for cme in simulation.cmes
        ):
            parts.append("DONKI")
        elif simulation.cmes:
            parts.append("Cone")
    solver_names = {
        "huxt": "SURF/HUXt",
        "hydro": "SURF/hydro",
        "huxt-pui": "SURF/HUXt_PUI",
        "hydro-pui": "SURF/hydro_PUI",
    }
    parts.append(solver_names.get(str(simulation.model.get("solver", "huxt")).lower(), "SURF/HUXt"))
    return "-".join(parts)


def style_timeseries_legends(axes, run_name, ambient_name=None):
    """Apply consistent names, size, and observation/model/ambient ordering."""
    for axis in np.atleast_1d(axes):
        handles, labels = axis.get_legend_handles_labels()
        if not handles:
            continue
        renamed = []
        for label in labels:
            lower = label.lower()
            if "ambient" in lower:
                renamed.append(ambient_name or label)
            elif "surf" in lower:
                renamed.append(run_name)
            else:
                renamed.append(label)
        ordered = sorted(
            zip(handles, renamed),
            key=lambda item: (
                0 if item[1].upper().startswith("OMNI") else
                2 if ambient_name is not None and item[1] == ambient_name else 1
            ),
        )
        axis.legend(
            [item[0] for item in ordered],
            [item[1] for item in ordered],
            fontsize=TIMESERIES_LEGEND_FONTSIZE,
        )


def format_datetime_axis_like_surf(fig, axes, times):
    """Format stacked datetime axes with identical ticks on every panel."""
    axes = np.atleast_1d(axes)
    starttime = times.iloc[0] if hasattr(times, "iloc") else times[0]
    endtime = times.iloc[-1] if hasattr(times, "iloc") else times[-1]
    # Matplotlib accepts Python, Pandas, and NumPy datetime values. Normalize
    # through its date converter so duration and year handling work for all
    # three representations (movie frame lines use numpy.datetime64).
    start_datetime = mdates.num2date(mdates.date2num(starttime))
    end_datetime = mdates.num2date(mdates.date2num(endtime))

    duration_days = max(
        0.0, (end_datetime - start_datetime).total_seconds() / 86400
    )
    major_interval_days = (
        max(1, int(np.ceil(duration_days / 6))) if duration_days >= 1 else None
    )
    for axis in axes:
        axis.set_xlim(start_datetime, end_datetime)
        if major_interval_days is None:
            major_locator = mdates.AutoDateLocator(minticks=3, maxticks=7)
        else:
            # Six intervals produce no more than about seven labels including
            # both ends.
            major_locator = mdates.DayLocator(interval=major_interval_days)
        axis.xaxis.set_major_locator(major_locator)
        axis.xaxis.set_minor_locator(
            mdates.DayLocator(interval=1)
            if major_interval_days is not None and major_interval_days > 1
            else mticker.NullLocator()
        )
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%d-%m"))

    fig.autofmt_xdate(rotation=0, ha="center")
    axes[-1].set_xlabel(
        f"DD-MM of {start_datetime.year}", fontsize=12, fontweight="bold"
    )

    return starttime, endtime


def _values(data):
    """Return plain numeric values from arrays that may carry units."""
    return np.asarray(data.value if hasattr(data, "value") else data, dtype=float)


def _solver_label(model):
    solver = str(getattr(model, "solver", "huxt"))
    return f"SURF-{solver}" if getattr(model, "compressible", False) else "SURF-HUXt"


def timeseries_figsize():
    """Return the standard figure size for all time-series plots."""
    return TIMESERIES_FIGSIZE


def sample_custom_timeseries(model, radius, lon):
    """Sample model grids at the nearest custom radius and longitude."""
    radial_index = int(np.argmin(np.abs(model.r - radius)))
    if model.lon.size == 1:
        longitude_index = 0
        lon_out = float(np.asarray(model.lon.value).reshape(-1)[0])
    else:
        longitude_index = int(np.argmin(np.abs(model.lon - lon)))
        lon_out = float(model.lon[longitude_index].value)

    series = {
        "time_days": np.asarray(model.time_out.to_value(u.day), dtype=float),
        "vsw": _values(model.v_grid[:, radial_index, longitude_index]),
        "r_out": float(model.r[radial_index].value),
        "lon_out": lon_out,
    }
    if hasattr(model, "time_init"):
        series["time"] = [(model.time_init + time).datetime for time in model.time_out]
    if hasattr(model, "b_grid"):
        series["bpol"] = _values(model.b_grid[:, radial_index, longitude_index])
    if getattr(model, "compressible", False):
        m_p = 1.6726e-27
        series["n"] = _values(model.rho_grid[:, radial_index, longitude_index]) / m_p / 1e6
        series["T"] = _values(model.temp_grid[:, radial_index, longitude_index])
    return series


def plot_custom_timeseries(model, radius, lon):
    """Plot a custom-coordinate time series using the observer plot style."""
    series = sample_custom_timeseries(model, radius, lon)
    times = series.get("time", series["time_days"])
    speed = series["vsw"]
    bpol = np.asarray(series.get("bpol", np.nan), dtype=float)
    has_bpol = np.isfinite(bpol).any()
    is_compressible = getattr(model, "compressible", False)

    n_panels = 1 + int(has_bpol) + (2 if is_compressible else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=timeseries_figsize(), sharex=True)
    axes = np.atleast_1d(axes)

    panel_index = 0
    axes[panel_index].plot(times, speed, "r", label=_solver_label(model))
    axes[panel_index].set_ylim(*SPEED_LIMITS_KMS)
    axes[panel_index].set_ylabel("V [km/s]")

    if has_bpol:
        panel_index += 1
        axes[panel_index].plot(times, np.sign(bpol), "r.", label=_solver_label(model))
        axes[panel_index].set_ylabel(r"B$_{\text{POL}}$")
        axes[panel_index].set_ylim(-1.1, 1.1)

    if is_compressible:
        panel_index += 1
        axes[panel_index].semilogy(times, series["n"], "r-", label=_solver_label(model))
        axes[panel_index].set_ylabel(r"n$_\text{P}$ [cm$^{-3}$]")
        axes[panel_index].set_ylim(0.101, 999)

        panel_index += 1
        axes[panel_index].semilogy(times, series["T"], "r-", label=_solver_label(model))
        axes[panel_index].set_ylabel(r"T [K]")
        axes[panel_index].set_ylim(1e4, 9.9e6)

    for axis in axes:
        axis.grid(True, alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend()
    for axis in axes[:-1]:
        axis.set_xticklabels([])

    if "time" in series:
        format_datetime_axis_like_surf(fig, axes, times)
    else:
        axes[-1].set_xlim(np.nanmin(times), np.nanmax(times))
        axes[-1].set_xlabel("Time (days)")
        for axis in axes[:-1]:
            axis.set_xlim(np.nanmin(times), np.nanmax(times))
    fig.subplots_adjust(left=0.10, bottom=0.14, right=0.98, top=0.95, hspace=0.05)
    frame = str(getattr(model, "frame", "model"))
    fig.suptitle(
        f"{_solver_label(model)} | r={series['r_out']:.1f} Rs | "
        f"fixed {frame} model lon={series['lon_out']:.1f} deg",
        fontsize=16,
    )
    return fig, axes if n_panels > 1 else axes[0]

