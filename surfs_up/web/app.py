"""Flask application factory for configuring and running SURF models."""

from __future__ import annotations

import datetime
import inspect
import io
import json
import logging
import math
import os
import pickle
import secrets
import tempfile
import threading
import time
import uuid
import webbrowser
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from flask import Flask, abort, jsonify, redirect, render_template, request, send_file, session, url_for

from surfs_up.core import (
    SimulationRequest,
    build_generated_code,
    format_datetime_axis_like_surf,
    full_run_name,
    plot_custom_timeseries,
    plot_radial as plot_radial_profile,
    run_generated_code,
    sample_custom_timeseries,
    style_timeseries_legends,
)
from surfs_up.jobs import enqueue, read_status
from surfs_up.web.insitu_cache import install_insitu_download_cache

_RUNS: OrderedDict[str, object] = OrderedDict()
_RUNS_LOCK = threading.Lock()
_RUN_PROGRESS: OrderedDict[str, str] = OrderedDict()
_RUN_PROGRESS_LOCK = threading.Lock()
_PLOT_LOCK = threading.Lock()


def _run_names(retained):
    simulation = retained.get("simulation")
    if not isinstance(simulation, SimulationRequest):
        return "SURF/HUXt", "SURF/HUXt (Ambient)"
    return (
        full_run_name(simulation),
        f"{full_run_name(simulation, ambient=True)} (Ambient)",
    )


def _replace_run_text(text, run_name, ambient_name):
    if text is None:
        return text
    value = str(text)
    if run_name in value or ambient_name in value:
        return value
    if "ambient" in value.lower():
        return ambient_name
    for short_name in ("SURF-HUXt_PUI", "SURF-HUXt", "SURF-hydro_PUI", "SURF-hydro", "SURF"):
        if short_name in value:
            return value.replace(short_name, run_name)
    return value


@contextmanager
def _movie_run_labels(run_name, ambient_name):
    """Style labels while SURF constructs and saves an animation internally."""
    from matplotlib.axes import Axes
    from matplotlib.text import Text

    original_legend = Axes.legend
    original_set_text = Text.set_text

    def labelled_legend(axis, *args, **kwargs):
        handles, labels = axis.get_legend_handles_labels()
        if handles and not args:
            renamed = [_replace_run_text(label, run_name, ambient_name) for label in labels]
            ordered = sorted(
                zip(handles, renamed),
                key=lambda item: (0 if item[1].upper().startswith("OMNI") else 2 if item[1] == ambient_name else 1),
            )
            args = ([item[0] for item in ordered], [item[1] for item in ordered])
        kwargs["fontsize"] = 12
        return original_legend(axis, *args, **kwargs)

    def labelled_text(artist, text):
        return original_set_text(
            artist, _replace_run_text(text, run_name, ambient_name)
        )

    Axes.legend = labelled_legend
    Text.set_text = labelled_text
    try:
        yield
    finally:
        Axes.legend = original_legend
        Text.set_text = original_set_text


class _ProgressPollLogFilter(logging.Filter):
    """Hide high-frequency progress polling from the development-server log."""

    def filter(self, record: logging.LogRecord) -> bool:
        return '"GET /run-progress/' not in record.getMessage()


class DonkiAccessError(RuntimeError):
    """Raised when NASA DONKI cannot be reached for CME data."""
_SURF_RUN_LOCK = threading.Lock()
# Solved models can be very large. The newest one is required for plots,
# time series, and movies; older models are superseded by the next run.
_MAX_RETAINED_RUNS = 1
_RUN_CACHE_DIR = Path(
    os.environ.get("SURFS_UP_RUN_CACHE_DIR", Path.home() / ".cache" / "surfs_up" / "runs")
)
_DONKI_URL = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/CMEAnalysis"
_PLOT_BODY_CHOICES = (
    ("MERCURY", "Mercury"), ("VENUS", "Venus"), ("EARTH", "Earth"),
    ("MARS", "Mars"), ("JUPITER", "Jupiter"), ("SATURN", "Saturn"),
    ("ACE", "ACE"), ("STA", "STEREO-A"), ("STB", "STEREO-B"),
    ("PSP", "Parker Solar Probe"), ("SOLO", "Solar Orbiter"),
    ("ULYSSES", "Ulysses"), ("NEW_HORIZONS", "New Horizons"),
)
_HORIZONS_TARGETS = {
    "URANUS": (799, "Uranus"),
    "NEW_HORIZONS": (-98, "New Horizons"),
}
_EPHEMERIS_STEP = "12H"


def _secret_key() -> str:
    """Return a stable signing key, preferring an explicit deployment secret."""
    configured = os.environ.get("SURFS_UP_SECRET_KEY")
    if configured:
        return configured
    path = Path.home() / ".cache" / "surfs_up" / "flask-secret-key"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        generated = secrets.token_urlsafe(48)
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(generated)
            return generated
        except FileExistsError:
            return path.read_text(encoding="utf-8").strip()
    except OSError:
        return secrets.token_urlsafe(48)


def _configure_animation_ffmpeg() -> None:
    """Point Matplotlib at the FFmpeg binary bundled by imageio-ffmpeg."""
    import imageio_ffmpeg
    import matplotlib

    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()


def _hydro_plot_with_calendar_date(plot_compressible):
    """Wrap SURF's hydro map plotter to show UTC date beside elapsed time."""

    def dated_plot(model, time, *args, **kwargs):
        result = plot_compressible(model, time, *args, **kwargs)
        figure = result[0]
        timestamp = (model.time_init + time).strftime("%Y-%m-%d %H:%M")
        for label in figure.texts:
            elapsed = label.get_text().strip()
            if elapsed.endswith(" days"):
                label.set_text(f"{elapsed} | {timestamp}")
                break
        return result

    return dated_plot


def _session_id() -> str:
    """Return the signed-cookie-backed identifier for the current browser session."""
    identifier = session.get("surf_session_id")
    if not identifier:
        identifier = uuid.uuid4().hex
        session["surf_session_id"] = identifier
    return str(identifier)


def _retain_model(
    model: object, simulation: SimulationRequest, ambient_model: object | None = None
) -> str:
    run_id = uuid.uuid4().hex
    retained = {
        "model": model,
        "ambient_model": ambient_model,
        "simulation": simulation,
        "owner_session_id": _session_id(),
    }
    with _RUNS_LOCK:
        _RUNS[run_id] = retained
        while len(_RUNS) > _MAX_RETAINED_RUNS:
            _RUNS.popitem(last=False)
    _write_run_cache(run_id, retained)
    return run_id


def _run_for(run_id: str) -> dict[str, object]:
    with _RUNS_LOCK:
        retained = _RUNS.get(run_id)
    if retained is None:
        retained = _read_run_cache(run_id)
    if retained is None:
        abort(404, "Run not found or no longer retained.")
    if not isinstance(retained, dict) or retained.get("owner_session_id") != _session_id():
        abort(404, "Run not found or no longer retained.")
    with _RUNS_LOCK:
        _RUNS[run_id] = retained
        while len(_RUNS) > _MAX_RETAINED_RUNS:
            _RUNS.popitem(last=False)
    return retained


def _model_for(run_id: str) -> object:
    return _run_for(run_id)["model"]


def _run_cache_path(run_id: str) -> Path:
    if not run_id.isalnum():
        abort(404)
    return _RUN_CACHE_DIR / f"{run_id}.pickle"


def _write_run_cache(run_id: str, retained: dict[str, object]) -> bool:
    try:
        _RUN_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _run_cache_path(run_id)
        with tempfile.NamedTemporaryFile(
            "wb", dir=_RUN_CACHE_DIR, delete=False, prefix=f"{run_id}.", suffix=".tmp"
        ) as handle:
            pickle.dump(retained, handle, protocol=pickle.HIGHEST_PROTOCOL)
            temp_path = Path(handle.name)
        temp_path.replace(path)
        _prune_run_cache()
        return True
    except Exception:
        # In-memory retention is still enough for local/single-worker use; a disk
        # cache is only needed when a deployment serves follow-up plot requests
        # from a different Python process.
        return False


def _read_run_cache(run_id: str) -> dict[str, object] | None:
    try:
        path = _run_cache_path(run_id)
        with path.open("rb") as handle:
            retained = pickle.load(handle)
        os.utime(path, None)
        return retained if isinstance(retained, dict) else {"model": retained, "simulation": None}
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _prune_run_cache() -> None:
    """Remove superseded models and abandoned partial pickle writes."""
    if not _RUN_CACHE_DIR.exists():
        return
    abandoned_before = time.time() - 60 * 60
    for path in _RUN_CACHE_DIR.glob("*.tmp"):
        try:
            if path.stat().st_mtime < abandoned_before:
                path.unlink(missing_ok=True)
        except FileNotFoundError:
            pass
    cached_runs = sorted(
        _RUN_CACHE_DIR.glob("*.pickle"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    cutoff = time.time() - 24 * 60 * 60
    for path in cached_runs[_MAX_RETAINED_RUNS:]:
        path.unlink(missing_ok=True)
    for path in cached_runs[:_MAX_RETAINED_RUNS]:
        if path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)


def _json_safe(value):
    """Return a JSON-safe copy of nested run metadata."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _set_run_progress(progress_id: str, message: str) -> None:
    """Store a short-lived status message for a run being processed."""
    if not progress_id:
        return
    progress_key = f"{_session_id()}:{progress_id}"
    with _RUN_PROGRESS_LOCK:
        _RUN_PROGRESS[progress_key] = message
        while len(_RUN_PROGRESS) > 16:
            _RUN_PROGRESS.popitem(last=False)


def _earth_latitude_at(model_time: datetime.datetime):
    """Return Earth's heliographic latitude without relying on sampled ephemeris data."""
    import astropy.units as u
    from sunpy.coordinates import sun

    return sun.B0(model_time).to(u.deg)


_AVERAGE_LATITUDE_BODIES = {
    "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
    "New Horizons",
}
_AVERAGE_LATITUDE_SPACECRAFT = {
    "STEREO-A": "STA",
    "STEREO-B": "STB",
    "Parker Solar Probe": "PSP",
    "Solar Orbiter": "SOLO",
}


def _average_body_latitude(
    body: str, start: datetime.datetime, duration_days: float
) -> float:
    """Return a time-sampled mean heliographic latitude for a Solar-System body."""
    import numpy as np
    import astropy.units as u
    from astropy.time import Time
    from sunpy.coordinates.ephemeris import get_body_heliographic_stonyhurst

    if body not in _AVERAGE_LATITUDE_BODIES | set(_AVERAGE_LATITUDE_SPACECRAFT):
        raise ValueError(f"Unsupported body: {body}")
    if not math.isfinite(duration_days) or duration_days <= 0:
        raise ValueError("Run duration must be positive.")

    # Daily samples, including both endpoints, resolve the smooth orbital latitude
    # variation without making long runs unnecessarily expensive.
    sample_count = max(2, int(math.ceil(duration_days)) + 1)
    offsets = np.linspace(0.0, duration_days, sample_count)
    times = Time(start) + offsets * u.day
    if body in _AVERAGE_LATITUDE_SPACECRAFT:
        from surf.surf import Observer

        coordinates = Observer(_AVERAGE_LATITUDE_SPACECRAFT[body], times)
        return float(np.mean(coordinates.lat.to_value(u.deg)))
    if body == "New Horizons":
        positions = _horizons_positions("NEW_HORIZONS", times[0], times[-1])
        return float(np.mean(np.rad2deg(positions["lat_rad"])))
    coordinates = get_body_heliographic_stonyhurst(body, times)
    return float(np.mean(coordinates.lat.to_value(u.deg)))


def _available_average_latitude_spacecraft(
    start: datetime.datetime, duration_days: float
) -> list[str]:
    """Return spacecraft whose local ephemerides cover the complete model run."""
    from astropy.time import Time
    from surf.surf import Observer

    times = Time([start, start + datetime.timedelta(days=duration_days)])
    available = []
    for label, observer_code in _AVERAGE_LATITUDE_SPACECRAFT.items():
        try:
            Observer(observer_code, times)
        except (KeyError, OSError, ValueError):
            continue
        available.append(label)
    return available


def _body_model_longitude_range(
    body: str, start: datetime.datetime, duration_days: float, nlon: int,
    dr_rs: float = 1.5,
) -> tuple[float, float, float]:
    """Return padded sidereal longitude and radial bounds for an observer."""
    import numpy as np
    import astropy.units as u
    from astropy.time import Time
    from surf.surf import Observer

    supported = {value for value, _label in _PLOT_BODY_CHOICES if value != "ACE"} | {"URANUS"}
    body_key = body.strip().upper()
    if body_key not in supported:
        raise ValueError(f"Unsupported body: {body}")
    if not math.isfinite(duration_days) or duration_days <= 0:
        raise ValueError("Run duration must be positive.")
    if nlon <= 0:
        raise ValueError("Longitude grid points must be positive.")
    if not math.isfinite(dr_rs) or dr_rs <= 0:
        raise ValueError("Radial grid spacing must be positive.")

    sample_count = max(2, int(math.ceil(duration_days * 4.0)) + 1)
    offsets = np.linspace(0.0, duration_days, sample_count)
    times = Time(start) + offsets * u.day
    if body_key in _HORIZONS_TARGETS:
        horizons = _horizons_positions(body_key, times[0], times[-1])
        times = Time(horizons["mjd"], format="mjd")
        observer_heeq_lon = np.asarray(horizons["lon_rad"], dtype=float)
        observer_radii = np.asarray(horizons["r_rs"], dtype=float)
    else:
        observer = Observer(body_key, times)
        observer_heeq_lon = (observer.lon_hae - Observer("EARTH", times).lon_hae).to_value(u.rad)
        observer_radii = observer.r.to_value(u.solRad)
    earth = Observer("EARTH", times)

    # Match surf_analysis.get_observer_timeseries for a sidereal model.
    earth_model_lon = earth.lon.to_value(u.rad) + (
        earth.lon_hae - earth.lon_hae[0]
    ).to_value(u.rad)
    observer_model_lon = earth_model_lon + observer_heeq_lon
    unwrapped = np.unwrap(observer_model_lon)
    if not np.all(np.isfinite(unwrapped)):
        raise ValueError(f"No ephemeris data are available for {body} over this run.")
    if not np.all(np.isfinite(observer_radii)):
        raise ValueError(f"No ephemeris data are available for {body} over this run.")
    r_max_rs = float(np.max(observer_radii)) + dr_rs

    margin = 360.0 / int(nlon)
    lower = math.degrees(float(np.min(unwrapped))) - margin
    upper = math.degrees(float(np.max(unwrapped))) + margin
    if upper - lower >= 360.0:
        return 0.0, 360.0, r_max_rs
    return lower % 360.0, upper % 360.0, r_max_rs


def _horizons_positions(target: str, start, stop):
    """Fetch one target using the same JPL Horizons adapter used for Uranus."""
    import surf.surf_analysis as surfA

    target_key = target.strip().upper().replace(" ", "_")
    naif_code, body_name = _HORIZONS_TARGETS[target_key]
    return surfA.get_horizons_body_for_SURF(
        start, stop, step=_EPHEMERIS_STEP,
        naif_code=naif_code, body_name=body_name,
    )


def _horizons_timeseries(model, target: str):
    """Sample a solved SURF model along a JPL Horizons trajectory."""
    import surf.surf_analysis as surfA

    start = model.time_init
    stop = model.time_init + model.time_out[-1]
    positions = _horizons_positions(target, start, stop)
    return surfA.get_SURF_at_position_HEEQ(
        model, positions["mjd"], positions["r_rs"], positions["lon_rad"]
    )


@contextmanager
def _horizons_plot_observers(model, surf_analysis, bodies):
    """Temporarily expose Horizons-only targets to SURF map/movie plotters."""
    selected = {str(body).upper() for body in bodies or ()}
    targets = selected & set(_HORIZONS_TARGETS)
    if not targets:
        yield
        return
    import numpy as np
    import astropy.units as u
    from astropy.time import Time

    model_times = Time(model.time_init + model.time_out)
    observers = {}
    for target in targets:
        positions = _horizons_positions(target, model_times[0], model_times[-1])
        source_times = Time(positions["mjd"], format="mjd")
        observers[target] = SimpleNamespace(
            r=np.interp(model_times.jd, source_times.jd, positions["r_rs"]) * u.solRad,
            lon=(np.interp(
                model_times.jd, source_times.jd, np.unwrap(positions["lon_rad"])
            ) % (2 * np.pi)) * u.rad,
            lat=np.interp(model_times.jd, source_times.jd, positions["lat_rad"]) * u.rad,
        )
    original_get_observer = model.get_observer
    original_styles = surf_analysis.observer_styles

    def get_observer(_model, body):
        key = str(body).upper().replace(" ", "_")
        return observers.get(key) or original_get_observer(body)

    def observer_styles():
        styles = original_styles()
        styles["NEW_HORIZONS"] = {"marker": "P", "color": "tab:purple"}
        return styles

    model.get_observer = MethodType(get_observer, model)
    surf_analysis.observer_styles = observer_styles
    try:
        yield
    finally:
        del model.get_observer
        surf_analysis.observer_styles = original_styles


def _model_defaults() -> dict[str, object]:
    """Return the same time-dependent defaults initialized by the Qt model tab."""
    import astropy.units as u
    import surf.surf_inputs as sin
    import surf.surf as s

    today = datetime.datetime.now(datetime.UTC).replace(tzinfo=None, microsecond=0)
    now = (
        today - datetime.timedelta(days=5)
    ).replace(tzinfo=None, microsecond=0)
    cr_num, cr_lon = sin.datetime2surfinputs(now)
    earth_latitude = _average_body_latitude("Earth", now, 10.0)
    surf_defaults = s.surf_constants()
    return {
        "default_start": now.strftime("%Y-%m-%dT%H:%M:%S"),
        "default_iswa_map_datetime": (now + datetime.timedelta(days=5)).strftime(
            "%Y-%m-%dT%H:%M"
        ),
        "default_cr_num": int(cr_num),
        "default_cr_lon": cr_lon.to_value(u.deg),
        "default_latitude": float(earth_latitude),
        "default_cme_density_pcc": surf_defaults["n_sw_21p5"].to_value(
            u.cm ** -3
        ),
        "default_cme_temperature_k": surf_defaults["T_sw_21p5"].to_value(u.K),
    }


def _float(name: str, default: float) -> float:
    value = request.form.get(name, "").strip()
    return float(value) if value else default


def _requested_plot_bodies():
    """Parse an optional comma-separated observer override from the query string."""
    if "bodies" not in request.args:
        return None
    value = request.args.get("bodies", "").strip()
    return [body.strip().upper() for body in value.split(",") if body.strip()]


def _default_plot_bodies(model) -> list[str]:
    """Return SURF's radius/date-dependent default bodies for a solved model."""
    import surf.surf_analysis as sa

    return [
        body
        for body in sa.get_planets_to_plot(model) + sa.get_spacecraft_to_plot(model)
        if body != "ACE"
    ]


def _default_insitu_source(model_time) -> str:
    """Prefer SWPC for runs within the rolling three-month real-time window."""
    if hasattr(model_time, "to_datetime"):
        model_time = model_time.to_datetime()
    if getattr(model_time, "tzinfo", None) is not None:
        model_time = model_time.replace(tzinfo=None)
    cutoff = datetime.datetime.now(datetime.UTC).replace(tzinfo=None) - datetime.timedelta(days=92)
    return "SWPC" if model_time >= cutoff else "OMNI"


def _available_plot_body_choices(model):
    """Return observers that enter the solved model's radial domain during the run."""
    import numpy as np

    outer_radius = model.r[-1]
    available = []
    for value, label in _PLOT_BODY_CHOICES:
        if value == "NEW_HORIZONS":
            import astropy.units as u

            if outer_radius > 20 * u.AU:
                available.append((value, label))
            continue
        try:
            radii = model.get_observer(value).r.to_value(outer_radius.unit)
        except Exception:
            continue
        if np.any(np.isfinite(radii) & (radii <= outer_radius.value)):
            available.append((value, label))
    return tuple(available)


def _save_uploaded_file(uploaded) -> Path:
    upload_dir = (
        Path(tempfile.gettempdir()) / "surfs_up_uploads" / _session_id()
    )
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded.filename).suffix
    path = upload_dir / f"{uuid.uuid4().hex}{suffix}"
    uploaded.save(path)
    return path


def _fetch_donki_cmes(
    start: datetime.datetime,
    duration_days: float,
    solver: str = "huxt",
    feature: str = "LE",
) -> list[dict[str, object]]:
    """Download and normalize DONKI cone CMEs for a model run interval."""
    end = start + datetime.timedelta(days=duration_days)
    feature = str(feature).strip().upper()
    if feature not in {"LE", "SH", "NULL"}:
        raise ValueError("DONKI feature must be LE, SH, or null.")
    query_params = {
        "startDate": start.date().isoformat(),
        "endDate": end.date().isoformat(),
        "mostAccurateOnly": "true",
        "catalog": "ALL",
    }
    # Older DONKI CME analyses predate the feature field. Omitting the filter is
    # how the API exposes those null/unspecified records.
    if feature != "NULL":
        query_params["feature"] = feature
    query = urlencode(query_params)
    try:
        with urlopen(f"{_DONKI_URL}?{query}", timeout=30) as response:
            analyses = json.load(response)
    except (URLError, TimeoutError) as exc:
        raise DonkiAccessError(
            "DONKI CME data could not be accessed. Check your network connection "
            "or try again later. You can run without DONKI data by unticking "
            "'Grab DONKI CMEs at run start'."
        ) from exc
    results = []
    for analysis in analyses:
        launch_text = analysis.get("time21_5")
        if not launch_text:
            continue
        launch = datetime.datetime.fromisoformat(
            str(launch_text).replace("Z", "+00:00")
        ).replace(tzinfo=None)
        if any(
            analysis.get(key) is None
            for key in ("longitude", "latitude", "speed", "halfAngle")
        ):
            continue
        results.append(
            {
                # ConeCME normalizes HEEQ longitude into the [0, 360) domain.
                "longitude": float(analysis["longitude"]) % 360.0,
                "latitude": float(analysis["latitude"]),
                "speed": float(analysis["speed"]),
                "width": 2 * float(analysis["halfAngle"]),
                "t_launch_day": (launch - start).total_seconds() / 86400,
                "t_launch_datetime": launch.strftime("%Y-%m-%d %H:%M:%S"),
                "thickness_rs": 0,
                "initial_height_rs": 21.5,
                "cme_expansion": False,
                "cme_fixed_duration": True,
                "fixed_duration_hr": 12,
                "profile_type": (
                    "sinusoidal" if str(solver).strip().lower() in {"hydro", "hydro-pui"} else "square"
                ),
                "plasma_mode": "Fraction of ambient",
                "density_fraction": 1,
                "temperature_fraction": 1,
                "source": "donki",
            }
        )
    # cone_dict_to_cme_list(), used by sin.get_DONKI_cme_list(), sorts by launch time.
    return sorted(results, key=lambda cme: float(cme["t_launch_day"]))


def _parse_cone_cmes(path: Path, model_start: datetime.datetime) -> list[dict[str, object]]:
    """Normalize a SURF cone2bc input file for the web CME editor."""
    import surf.surf_inputs as sin
    from astropy.time import Time

    results = []
    for cone in sin.import_cone2bc_parameters(str(path)).values():
        launch = Time(cone["ldates"]).to_datetime().replace(tzinfo=None)
        results.append(
            {
                "longitude": float(cone.get("lon", 0)),
                "latitude": float(cone.get("lat", 0)),
                "speed": float(cone.get("vcld", 800)),
                "width": float(2 * cone.get("rmajor", 30)),
                "t_launch_day": (launch - model_start).total_seconds() / 86400,
                "t_launch_datetime": launch.strftime("%Y-%m-%d %H:%M:%S"),
                "source": "cone_file",
            }
        )
    return sorted(results, key=lambda cme: float(cme["t_launch_day"]))


def _example_input_path(pattern: str, missing_message: str) -> Path:
    import surf

    examples = Path(surf.__file__).resolve().parent / "data" / "example_inputs"
    matches = sorted(examples.glob(pattern))
    if not matches:
        raise ValueError(missing_message)
    return matches[0]


def _parse_wsa_start_time(filepath: Path):
    """Extract WSA map time from FITS metadata when available, else filename."""
    import re

    from astropy.io import fits

    filepath = Path(filepath)

    try:
        header = fits.getheader(filepath)
        for key in ("DATE-OBS", "DATE_OBS", "DATE", "MAPDATE"):
            if key in header:
                value = str(header[key]).strip()
                for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d"):
                    try:
                        return datetime.datetime.strptime(value[:19], fmt)
                    except ValueError:
                        continue
    except Exception:
        pass

    name = filepath.name
    match = re.search(r"(\d{4}-\d{2}-\d{2})T(\d{2})Z", name)
    if match:
        return datetime.datetime.strptime(
            f"{match.group(1)}T{match.group(2)}", "%Y-%m-%dT%H"
        )

    match = re.search(r"(\d{8})(\d{2})", name)
    if match:
        return datetime.datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H")

    return None


def _parse_cortom_start_time(filepath: Path):
    """Extract CorTom map time from filename."""
    import re

    filepath = Path(filepath)
    match = re.search(r"(\d{14})", filepath.name)
    if match:
        return datetime.datetime.strptime(match.group(1), "%Y%m%d%H%M%S")
    return None


def _ambient_file_start_time(source: str, filepath: Path):
    if source == "wsa":
        return _parse_wsa_start_time(filepath)
    if source == "cortom":
        return _parse_cortom_start_time(filepath)
    return None


def _iswa_map_datetime(value: str, fallback: str) -> datetime.datetime:
    """Parse the ISWA WSA map date/time control.

    Accept both date-only values from older forms/configurations and
    ``datetime-local`` values from the current web interface.
    """
    text = (value or fallback or "").strip().replace(" ", "T")
    if not text:
        return datetime.datetime.now(datetime.UTC).replace(tzinfo=None, microsecond=0)
    if "T" not in text:
        text = f"{text}T23:59:59"
    return datetime.datetime.fromisoformat(text)


def _draw_speed_map(ax, speed_map, longitudes, latitudes, extraction_latitude, title):
    import astropy.units as u
    import numpy as np

    speed_values = (
        speed_map.to_value(u.km / u.s)
        if hasattr(speed_map, "to_value")
        else np.asarray(speed_map)
    )
    lon_values = (
        longitudes.to_value(u.deg)
        if hasattr(longitudes, "to_value")
        else np.rad2deg(np.asarray(longitudes))
    )
    lat_values = (
        latitudes.to_value(u.deg)
        if hasattr(latitudes, "to_value")
        else np.rad2deg(np.asarray(latitudes))
    )
    speed_values = np.asarray(speed_values)
    if speed_values.shape == (len(lon_values), len(lat_values)):
        speed_values = speed_values.T
    if speed_values.shape != (len(lat_values), len(lon_values)):
        raise ValueError(
            "Speed map dimensions do not match its longitude and latitude coordinates."
        )

    image = ax.pcolormesh(
        lon_values,
        lat_values,
        speed_values,
        shading="auto",
        cmap="viridis",
    )
    ax.axhline(
        extraction_latitude,
        color="red",
        linewidth=1.8,
        linestyle="--",
        label=f"Extracted latitude: {extraction_latitude:.1f}°",
    )
    ax.set_xlim(float(np.nanmin(lon_values)), float(np.nanmax(lon_values)))
    ax.set_ylim(float(np.nanmin(lat_values)), float(np.nanmax(lat_values)))
    ax.set_ylabel("Latitude [deg]")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.figure.colorbar(image, ax=ax, label="Speed [km/s]")


def _ambient_preview_figure():
    import astropy.units as u
    import matplotlib
    import numpy as np
    import surf.surf_inputs as sin

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = request.form.get("ambient_source", "user_specified")
    latitude = _float("latitude", 0.0) * u.deg
    include_bpol = "include_bpol" in request.form
    solver = request.form.get("solver", "huxt").strip().lower()
    acc_profile = "huxt" if solver.startswith("huxt") else "parker"

    def plot_mas():
        cr_num = int(_float("mas_cr_num", 2000))
        source_radius_rs = _float("mas_source_radius_rs", 30.0)
        map_to_inner = "mas_decelerate" in request.form
        speed_map, map_longitudes, map_latitudes = sin.get_MAS_vr_map(cr_num)
        v_orig = sin.get_MAS_long_profile(cr_num, latitude)
        if include_bpol:
            b_orig = sin.get_MAS_br_long_profile(cr_num, latitude)
            if len(b_orig) != len(v_orig):
                b_lon = np.linspace(0.0, 360.0, len(b_orig), endpoint=False)
                v_lon = np.linspace(0.0, 360.0, len(v_orig), endpoint=False)
                b_orig = np.interp(v_lon, b_lon, np.asarray(b_orig), period=360.0)
            if map_to_inner:
                mapped = sin.map_v_boundary_inwards(
                    v_orig,
                    source_radius_rs * u.solRad,
                    21.5 * u.solRad,
                    acc_profile=acc_profile,
                    b_orig=b_orig,
                )
                if isinstance(mapped, tuple):
                    v_mapped, b_mapped = mapped
                else:
                    v_mapped = mapped
                    b_mapped = np.ones(len(v_orig)) * np.nan
            else:
                v_mapped = v_orig
                b_mapped = b_orig
        else:
            b_orig = None
            if map_to_inner:
                v_mapped = sin.map_v_boundary_inwards(
                    v_orig,
                    source_radius_rs * u.solRad,
                    21.5 * u.solRad,
                    acc_profile=acc_profile,
                )
            else:
                v_mapped = v_orig

        carr_lon = np.linspace(0.0, 360.0, len(v_orig), endpoint=False)
        if include_bpol:
            fig, (ax_map, ax_v, ax_b) = plt.subplots(3, 1, figsize=(10, 12))
        else:
            fig, (ax_map, ax_v) = plt.subplots(2, 1, figsize=(10, 9))
        _draw_speed_map(
            ax_map,
            speed_map,
            map_longitudes,
            map_latitudes,
            latitude.value,
            f"MAS speed map | CR {cr_num}",
        )
        ax_map.set_xlabel("Carrington longitude [deg]")
        ax_v.plot(carr_lon, v_orig.to_value(u.km / u.s), linewidth=1.5, label=f"Original at {source_radius_rs:g} Rs")
        ax_v.plot(
            carr_lon,
            v_mapped.to_value(u.km / u.s),
            linewidth=1.5,
            linestyle="--",
            label=(
                "Mapped to 21.5 Rs" if map_to_inner else "Original (no deceleration mapping)"
            ),
        )
        ax_v.set_xlim(0.0, 360.0)
        ax_v.set_ylabel("Vin [km/s]")
        ax_v.set_title(f"MAS boundary profiles | CR {cr_num} | lat {latitude.value:.1f} deg")
        ax_v.grid(True, alpha=0.3)
        ax_v.legend()
        if include_bpol:
            ax_b.plot(carr_lon, np.asarray(b_orig), linewidth=1.5, label=f"Original bpol at {source_radius_rs:g} Rs")
            ax_b.plot(
                carr_lon,
                np.asarray(b_mapped),
                linewidth=1.5,
                linestyle="--",
                label=(
                    "Mapped bpol to 21.5 Rs" if map_to_inner else "Original bpol (no deceleration mapping)"
                ),
            )
            ax_b.set_xlim(0.0, 360.0)
            ax_b.set_xlabel("Carrington longitude [deg]")
            ax_b.set_ylabel("bpol")
            ax_b.grid(True, alpha=0.3)
            ax_b.legend()
        else:
            ax_v.set_xlabel("Carrington longitude [deg]")
        fig.tight_layout()
        return fig

    def plot_file_source(
        title: str,
        speed_map_title: str,
        path: Path,
        source_radius_rs: float,
        profile_loader,
        br_profile_loader,
        speed_map_loader,
        decelerate_key: str,
        reduction_key: str | None = None,
    ):
        map_to_inner = decelerate_key in request.form
        apply_speed_reduction = reduction_key is not None and reduction_key in request.form
        speed_map, map_longitudes, map_latitudes = speed_map_loader(path)
        v_orig = profile_loader(path, latitude)
        if apply_speed_reduction:
            longitude = np.linspace(0.0, 2.0 * np.pi, len(v_orig), endpoint=False) * u.rad
            mapper = sin.map_v_inwards if solver.startswith("huxt") else sin.map_v_inwards_parker
            wsa_reduction = mapper(
                v_orig,
                215.0 * u.solRad,
                longitude,
                21.5 * u.solRad,
            )
            v_reduced = wsa_reduction[0]
        else:
            v_reduced = v_orig

        include_bpol_plot = include_bpol and (br_profile_loader is not None)
        if include_bpol_plot:
            b_orig = br_profile_loader(path, latitude)
            if map_to_inner:
                mapped = sin.map_v_boundary_inwards(
                    v_reduced,
                    source_radius_rs * u.solRad,
                    21.5 * u.solRad,
                    acc_profile=acc_profile,
                    b_orig=b_orig,
                )
                if isinstance(mapped, tuple):
                    v_mapped, b_mapped = mapped
                else:
                    v_mapped = mapped
                    b_mapped = np.ones(len(v_orig)) * np.nan
            else:
                v_mapped = v_reduced
                b_mapped = b_orig
        else:
            if map_to_inner:
                v_mapped = sin.map_v_boundary_inwards(
                    v_reduced,
                    source_radius_rs * u.solRad,
                    21.5 * u.solRad,
                    acc_profile=acc_profile,
                )
            else:
                v_mapped = v_reduced

        carr_lon = np.linspace(0.0, 360.0, len(v_orig), endpoint=False)
        include_speed_map = speed_map_loader is not None
        if include_speed_map and include_bpol_plot:
            fig, (ax_map, ax_v, ax_b) = plt.subplots(3, 1, figsize=(10, 12))
        elif include_speed_map:
            fig, (ax_map, ax_v) = plt.subplots(2, 1, figsize=(10, 9))
        elif include_bpol_plot:
            fig, (ax_v, ax_b) = plt.subplots(2, 1, sharex=True)
        else:
            fig, ax_v = plt.subplots()
        if include_speed_map:
            _draw_speed_map(
                ax_map,
                speed_map,
                map_longitudes,
                map_latitudes,
                latitude.value,
                speed_map_title,
            )
            ax_map.set_xlabel("Carrington longitude [deg]")
        ax_v.plot(
            carr_lon,
            v_orig.to_value(u.km / u.s),
            linewidth=1.5,
            label=f"Original at {source_radius_rs:.1f} Rs",
        )
        if apply_speed_reduction:
            ax_v.plot(
                carr_lon,
                v_reduced.to_value(u.km / u.s),
                linewidth=1.5,
                linestyle="-.",
                label="WSA speed reduction: 215 to 21.5 Rs (longitude unchanged)",
            )
        ax_v.plot(
            carr_lon,
            v_mapped.to_value(u.km / u.s),
            linewidth=1.5,
            linestyle="--",
            label=(
                "Mapped to 21.5 Rs"
                if map_to_inner
                else ("Speed-reduced boundary" if apply_speed_reduction else "Original (no deceleration mapping)")
            ),
        )
        ax_v.set_xlim(0.0, 360.0)
        ax_v.set_ylabel("Vin [km/s]")
        ax_v.grid(True, alpha=0.3)
        ax_v.legend()
        if include_bpol_plot:
            ax_b.plot(
                carr_lon,
                np.asarray(b_orig),
                linewidth=1.5,
                label=f"Original bpol at {source_radius_rs:.1f} Rs",
            )
            ax_b.plot(
                carr_lon,
                np.asarray(b_mapped),
                linewidth=1.5,
                linestyle="--",
                label=(
                    "Mapped bpol to 21.5 Rs" if map_to_inner else "Original bpol (no deceleration mapping)"
                ),
            )
            ax_b.set_xlim(0.0, 360.0)
            ax_b.set_xlabel("Carrington longitude [deg]")
            ax_b.set_ylabel("bpol")
            ax_b.grid(True, alpha=0.3)
            ax_b.legend()
        else:
            ax_v.set_xlabel("Carrington longitude [deg]")
        fig.tight_layout()
        return fig

    if source == "mas":
        return plot_mas()
    if source == "wsa":
        uploaded = request.files.get("wsa_file")
        path = _save_uploaded_file(uploaded) if uploaded and uploaded.filename else _example_input_path(
            "**/*.fits",
            "Upload a WSA input file.",
        )
        return plot_file_source(
            "WSA boundary profiles",
            f"WSA speed map | {Path(path).name}",
            path,
            _float("wsa_source_radius_rs", 21.5),
            sin.get_WSA_long_profile,
            sin.get_WSA_br_long_profile,
            lambda selected_path: sin.get_WSA_maps(selected_path)[:3],
            "wsa_decelerate",
            "wsa_speed_reduction",
        )
    if source == "wsa_iswa":
        iswa_fallback = request.form.get("start_datetime", "")
        iswa_value = request.form.get("iswa_map_date", "")
        required_for = _iswa_map_datetime(
            iswa_value,
            iswa_fallback,
        )
        path = sin.get_WSA_from_ISWA(required_for)
        return plot_file_source(
            "WSA boundary profiles",
            f"WSA speed map | {Path(path).name}",
            Path(path),
            _float("iswa_source_radius_rs", 21.5),
            sin.get_WSA_long_profile,
            sin.get_WSA_br_long_profile,
            lambda selected_path: sin.get_WSA_maps(selected_path)[:3],
            "iswa_decelerate",
            "iswa_speed_reduction",
        )
    if source == "cortom":
        uploaded = request.files.get("cortom_file")
        path = _save_uploaded_file(uploaded) if uploaded and uploaded.filename else _example_input_path(
            "**/*.dat",
            "Upload a CORTOM input file.",
        )
        return plot_file_source(
            "CorTom boundary profiles",
            f"CorTom speed map | {Path(path).name}",
            path,
            _float("cortom_source_radius_rs", 8.0),
            sin.get_CorTom_long_profile,
            None,
            sin.get_CorTom_vr_map,
            "cortom_decelerate",
            None,
        )
    raise ValueError("Select MAS, WSA, WSA (ISWA), or CorTom before plotting.")


def _request_from_form() -> SimulationRequest:
    start = request.form.get("start_datetime") or datetime.datetime.now(
        datetime.UTC
    ).strftime("%Y-%m-%d %H:%M:%S")
    source = request.form.get("ambient_source", "user_specified")
    speed = _float("speed_kms", 400.0)
    ambient = {"source": source}
    if source == "user_specified":
        profile_text = request.form.get("speed_profile", "").strip()
        ambient["speed_profile_kms"] = (
            [float(value) for value in profile_text.split(",") if value.strip()]
            if profile_text
            else [speed] * 128
        )
    elif source == "mas":
        ambient.update(
            cr_num=int(_float("mas_cr_num", 2300)),
            source_radius_rs=_float("mas_source_radius_rs", 30.0),
            decelerate_to_inner_boundary="mas_decelerate" in request.form,
        )
        if "mas_use_map_time" in request.form:
            from sunpy.coordinates import sun

            map_time = sun.carrington_rotation_time(
                float(_float("mas_cr_num", 2300))
            ).to_datetime()
            if hasattr(map_time, "item"):
                map_time = map_time.item()
            if map_time.tzinfo is not None:
                map_time = map_time.replace(tzinfo=None)
            start = (map_time - datetime.timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
    elif source == "wsa":
        uploaded = request.files.get("wsa_file")
        if uploaded and uploaded.filename:
            path = _save_uploaded_file(uploaded)
        else:
            path = _example_input_path("**/*.fits", "Upload a WSA input file.")
        ambient.update(
            filepath=str(path),
            source_radius_rs=_float("wsa_source_radius_rs", 21.5),
            decelerate_to_inner_boundary="wsa_decelerate" in request.form,
            apply_wsa_speed_reduction="wsa_speed_reduction" in request.form,
        )
        if "wsa_use_map_time" in request.form:
            map_time = _ambient_file_start_time(source, path)
            if map_time is not None:
                start = (map_time - datetime.timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
    elif source == "wsa_iswa":
        iswa_datetime = _iswa_map_datetime(
            request.form.get("iswa_map_date", ""),
            start,
        )
        if "iswa_use_model_start" in request.form:
            start = (iswa_datetime - datetime.timedelta(days=5)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ambient.update(
            source_radius_rs=_float("iswa_source_radius_rs", 21.5),
            decelerate_to_inner_boundary="iswa_decelerate" in request.form,
            apply_wsa_speed_reduction="iswa_speed_reduction" in request.form,
            iswa_map_datetime=iswa_datetime.isoformat(),
        )
    elif source == "cortom":
        uploaded = request.files.get("cortom_file")
        if uploaded and uploaded.filename:
            path = _save_uploaded_file(uploaded)
        else:
            path = _example_input_path("**/*.dat", "Upload a CORTOM input file.")
        ambient.update(
            filepath=str(path),
            source_radius_rs=_float("cortom_source_radius_rs", 8.0),
            decelerate_to_inner_boundary="cortom_decelerate" in request.form,
        )
        if "cortom_use_map_time" in request.form:
            map_time = _ambient_file_start_time(source, path)
            if map_time is not None:
                start = (map_time - datetime.timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")
    elif source == "insitu_backmapped":
        ambient["mode"] = request.form.get("insitu_mode", "forecast")
        spacecraft = request.form.get("insitu_spacecraft", "OMNI")
        ambient["spacecraft"] = (
            spacecraft if spacecraft in {"OMNI", "SWPC", "STEREO-A"} else "OMNI"
        )
        ambient["forecast_datetime"] = request.form.get("omni_forecast_datetime", "")
        requested_icme_list = request.form.get("omni_icme_list", "")
        allowed_icme_lists = (
            {"STEREO-A", "None"}
            if ambient["spacecraft"] == "STEREO-A"
            else {"CaneRichardson", "DONKI", "None"}
        )
        ambient["icme_list"] = (
            requested_icme_list
            if requested_icme_list in allowed_icme_lists
            else ("STEREO-A" if ambient["spacecraft"] == "STEREO-A" else "None")
        )
        pre_icme_buffer_days = _float("insitu_pre_icme_buffer_days", 0.2)
        post_icme_buffer_days = _float("insitu_post_icme_buffer_days", 1.0)
        if pre_icme_buffer_days < 0 or post_icme_buffer_days < 0:
            raise ValueError("ICME buffers must be zero or greater.")
        ambient["pre_icme_buffer_days"] = pre_icme_buffer_days
        ambient["post_icme_buffer_days"] = post_icme_buffer_days
        donki_min_quality = int(_float("donki_icme_min_quality", 1))
        if donki_min_quality not in {-1, 0, 1, 2}:
            raise ValueError("Minimum DONKI ICME quality must be -1, 0, 1, or 2.")
        ambient["donki_icme_min_quality"] = donki_min_quality
    elif source == "omni":
        ambient["use_215_inner_boundary"] = "use_215_inner_boundary" in request.form
        ambient["icme_list"] = request.form.get("omni_icme_list", "None")
        pre_icme_buffer_days = _float("insitu_pre_icme_buffer_days", 0.2)
        post_icme_buffer_days = _float("insitu_post_icme_buffer_days", 1.0)
        if pre_icme_buffer_days < 0 or post_icme_buffer_days < 0:
            raise ValueError("ICME buffers must be zero or greater.")
        ambient["pre_icme_buffer_days"] = pre_icme_buffer_days
        ambient["post_icme_buffer_days"] = post_icme_buffer_days
        donki_min_quality = int(_float("donki_icme_min_quality", 1))
        if donki_min_quality not in {-1, 0, 1, 2}:
            raise ValueError("Minimum DONKI ICME quality must be -1, 0, 1, or 2.")
        ambient["donki_icme_min_quality"] = donki_min_quality

    cmes_text = request.form.get("cmes_json", "").strip()
    cmes = json.loads(cmes_text) if cmes_text else []
    if not isinstance(cmes, list):
        raise ValueError("CME JSON must contain a list of CME objects.")
    grab_donki_at_run_start = (
        "grab_donki_at_run_start" in request.form
        and source != "omni"
        and abs(_float("rmin", 21.5) - 21.5) <= 1.0e-9
    )
    if grab_donki_at_run_start:
        # The generated script performs the DONKI request immediately before solving.
        # Discard the editor list so checked means replace, not merge.
        cmes = []
    cone_file = request.files.get("cone_file")
    if cone_file and cone_file.filename:
        import numpy as np

        cone_path = _save_uploaded_file(cone_file)
        model_start = datetime.datetime.fromisoformat(start.replace("T", " "))
        for cone in _parse_cone_cmes(cone_path, model_start):
            cmes.append(
                {
                    **cone,
                    "thickness_rs": 0,
                    "initial_height_rs": 21.5,
                    "cme_expansion": False,
                    "cme_fixed_duration": True,
                    "fixed_duration_hr": 12,
                    "profile_type": "square",
                    "plasma_mode": "Fraction of ambient",
                    "density_fraction": 1,
                    "temperature_fraction": 1,
                    "cme_density_pcc": np.nan,
                    "cme_temperature_k": np.nan,
                }
            )
    simtime_days = _float("simtime_days", 10.0)
    return SimulationRequest.from_mappings(
        {
            "solver": request.form.get("solver", "huxt"),
            "rmin": _float("rmin", 21.5),
            "rmax": _float("rmax", 240.0),
            "lon_min": _float("lon_min", 0.0),
            "lon_max": _float("lon_max", 360.0),
            "latitude": _float("latitude", 0.0),
            "is_1d": "is_1d" in request.form,
            "frame": request.form.get("frame", "synodic"),
            "include_bpol": "include_bpol" in request.form,
            "track_cmes": "track_cmes" in request.form,
            "run_ambient_solution": "run_ambient_solution" in request.form,
            "grab_donki_at_run_start": grab_donki_at_run_start,
            "donki_cme_defaults": {
                "feature": request.form.get("donki_feature", "LE").strip().upper(),
                "thickness_rs": _float("donki_thickness_rs", 0.0),
                "initial_height_rs": _float("donki_initial_height_rs", 21.5),
                "cme_expansion": "donki_cme_expansion" in request.form,
                "cme_fixed_duration": "donki_cme_fixed_duration" in request.form,
                "fixed_duration_hr": _float("donki_fixed_duration_hr", 12.0),
                "profile_type": request.form.get(
                    "donki_profile_type",
                    "sinusoidal" if request.form.get("solver") in {"hydro", "hydro-pui"} else "square",
                ),
                "plasma_mode": request.form.get("donki_plasma_mode", "Fraction of ambient"),
                "density_fraction": _float("donki_density_fraction", 1.0),
                "temperature_fraction": _float("donki_temperature_fraction", 1.0),
                "cme_density_pcc": _float("donki_cme_density_pcc", 100.0),
                "cme_temperature_k": _float("donki_cme_temperature_k", 100000.0),
            },
            "streak_lines_enabled": "streak_lines_enabled" in request.form,
            "streak_spacing_deg": _float("streak_spacing_deg", 10.0),
            "simtime_days": simtime_days,
            "dr_rs": _float("dr_rs", 1.5),
            "nlon": int(_float("nlon", 128)),
            "vmax_kms": _float("vmax_kms", 3000.0),
            "dt_scale": _float(
                "dt_scale", max(1, round(4.0 * simtime_days / 10.0))
            ),
            "chunked_solve": "chunked_solve" in request.form,
            "chunk_size_days": _float("chunk_size_days", 3.0),
            "gamma": _float("gamma", 1.5),
            "start_datetime": start.replace("T", " "),
            "cr_num": int(_float("cr_num", 2300)),
            "cr_lon_init_deg": _float("cr_lon_init_deg", 0.0),
        },
        ambient,
        cmes,
    )


def create_app(config: dict | None = None) -> Flask:
    """Create an app suitable for local use or a PythonAnywhere WSGI file."""
    install_insitu_download_cache()
    werkzeug_logger = logging.getLogger("werkzeug")
    if not any(
        isinstance(log_filter, _ProgressPollLogFilter)
        for log_filter in werkzeug_logger.filters
    ):
        werkzeug_logger.addFilter(_ProgressPollLogFilter())
    app = Flask(__name__)
    app.config.from_mapping(
        MAX_CONTENT_LENGTH=1_000_000,
        RUN_JOBS_SYNCHRONOUS=False,
        SECRET_KEY=_secret_key(),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    if config:
        app.config.update(config)
    if app.config["TESTING"] and (
        config is None or "RUN_JOBS_SYNCHRONOUS" not in config
    ):
        app.config["RUN_JOBS_SYNCHRONOUS"] = True
    if not app.config["TESTING"]:
        _prune_run_cache()

    @app.before_request
    def establish_browser_session():
        """Ensure every visitor receives an isolated signed session identifier."""
        _session_id()

    @app.get("/model-coordinates")
    def model_coordinates():
        """Convert between model UTC time and Carrington coordinates."""
        import astropy.units as u
        import surf.surf_inputs as sin
        from sunpy.coordinates import sun

        datetime_text = request.args.get("datetime", "").strip()
        if datetime_text:
            model_time = datetime.datetime.fromisoformat(datetime_text.replace("T", " "))
        else:
            cr_num = float(request.args["cr_num"])
            cr_lon = float(request.args["cr_lon"])
            cr_fraction = cr_num + ((360.0 - cr_lon) / 360.0)
            model_time = sun.carrington_rotation_time(cr_fraction).to_datetime()
            if hasattr(model_time, "item"):
                model_time = model_time.item()
            if model_time.tzinfo is not None:
                model_time = model_time.replace(tzinfo=None)

        cr_num, cr_lon = sin.datetime2surfinputs(model_time)
        earth_latitude = _earth_latitude_at(model_time)
        return jsonify(
            {
                "datetime": model_time.strftime("%Y-%m-%dT%H:%M:%S"),
                "cr_num": int(cr_num),
                "cr_lon": float(cr_lon.to_value(u.deg)),
                "earth_latitude": float(
                    earth_latitude.to_value(u.deg)
                    if hasattr(earth_latitude, "to_value")
                    else earth_latitude
                ),
            }
        )

    @app.get("/average-body-latitude")
    def average_body_latitude():
        """Calculate a body's mean heliographic latitude over a model run."""
        try:
            start = datetime.datetime.fromisoformat(
                request.args["datetime"].strip().replace("T", " ")
            )
            duration_days = float(request.args.get("duration", "10"))
            requested_body = request.args.get("body", "Earth").strip()
            body_lookup = {
                name.lower(): name
                for name in _AVERAGE_LATITUDE_BODIES | set(_AVERAGE_LATITUDE_SPACECRAFT)
            }
            body = body_lookup.get(requested_body.lower(), requested_body)
            latitude = _average_body_latitude(body, start, duration_days)
        except (KeyError, TypeError, ValueError) as error:
            abort(400, str(error))
        return jsonify({"body": body, "average_latitude": latitude})

    @app.get("/latitude-body-choices")
    def latitude_body_choices():
        """Return spacecraft with ephemeris coverage for the requested model interval."""
        try:
            start = datetime.datetime.fromisoformat(
                request.args["datetime"].strip().replace("T", " ")
            )
            duration_days = float(request.args.get("duration", "10"))
            if not math.isfinite(duration_days) or duration_days <= 0:
                raise ValueError("Run duration must be positive.")
            spacecraft = _available_average_latitude_spacecraft(start, duration_days)
        except (KeyError, TypeError, ValueError) as error:
            abort(400, str(error))
        return jsonify({"spacecraft": spacecraft})

    @app.get("/body-longitude-range")
    def body_longitude_range():
        """Calculate the sidereal model longitude arc needed for an observer."""
        try:
            start = datetime.datetime.fromisoformat(
                request.args["datetime"].strip().replace("T", " ")
            )
            duration_days = float(request.args.get("duration", "10"))
            nlon = int(request.args.get("nlon", "128"))
            dr_rs = float(request.args.get("dr_rs", "1.5"))
            body = request.args.get("body", "Earth").strip()
            lon_min, lon_max, r_max_rs = _body_model_longitude_range(
                body, start, duration_days, nlon, dr_rs
            )
        except (KeyError, TypeError, ValueError) as error:
            abort(400, str(error))
        return jsonify({
            "body": body, "lon_min": lon_min, "lon_max": lon_max,
            "r_max_rs": r_max_rs,
        })

    @app.post("/ambient-file-time")
    def ambient_file_time():
        """Infer the start time from a selected ambient file."""
        source = request.form.get("source", "")
        uploaded = request.files.get("file")
        if not uploaded or not uploaded.filename:
            abort(400, "Upload a file to infer its timestamp.")

        path = _save_uploaded_file(uploaded)
        map_time = _ambient_file_start_time(source, path)
        if map_time is None:
            return jsonify({"datetime": None})
        return jsonify({"datetime": map_time.strftime("%Y-%m-%dT%H:%M:%S")})

    @app.post("/generated-code")
    def generated_code():
        """Return generated Python code for the current form state."""
        try:
            simulation = _request_from_form()
            return jsonify({"code": build_generated_code(simulation)})
        except (DonkiAccessError, json.JSONDecodeError, TypeError, ValueError) as exc:
            abort(400, str(exc))

    @app.post("/cone-cmes")
    def cone_cmes():
        """Parse an uploaded cone2bc list for immediate display in the editor."""
        uploaded = request.files.get("file")
        if not uploaded or not uploaded.filename:
            abort(400, "Select a Cone CME list to load.")
        try:
            start = datetime.datetime.fromisoformat(
                request.form["start"].strip().replace("T", " ")
            )
            return jsonify(_parse_cone_cmes(_save_uploaded_file(uploaded), start))
        except (KeyError, TypeError, ValueError) as exc:
            abort(400, f"Could not read Cone CME list: {exc}")

    @app.route("/", methods=["GET", "POST"])
    def index():
        context = {
            "code": None,
            "error": None,
            "result": None,
            "run_id": None,
            "show_movies": False,
            "show_code_dialog": False,
            "plot_body_choices": _PLOT_BODY_CHOICES,
            "default_plot_bodies": [],
            "default_insitu_source": "SWPC",
            "ambient_model_available": False,
        }
        requested_run_id = request.args.get("run_id", "")
        if requested_run_id:
            status = read_status(requested_run_id)
            if (
                status is None
                or status.get("owner_session_id") != _session_id()
                or status.get("state") not in {"completed", "failed"}
            ):
                abort(404, "Run not found or not yet complete.")
            context["result"] = SimpleNamespace(
                success=status["state"] == "completed",
                message=status.get("message", ""),
                output=status.get("output", ""),
            )
            if context["result"].success:
                context["run_id"] = requested_run_id
                context["show_movies"] = bool(status.get("show_movies", False))
                retained_run = _run_for(requested_run_id)
                retained_model = retained_run["model"]
                context["ambient_model_available"] = (
                    retained_run.get("ambient_model") is not None
                )
                context["default_plot_bodies"] = _default_plot_bodies(retained_model)
                context["plot_body_choices"] = _available_plot_body_choices(retained_model)
                context["default_insitu_source"] = _default_insitu_source(
                    retained_model.time_init
                )
        if request.method == "POST":
            try:
                simulation = _request_from_form()
                context["code"] = build_generated_code(simulation)
                action = request.form.get("action")
                context["show_code_dialog"] = action == "preview"
                if action == "run":
                    if not app.config["RUN_JOBS_SYNCHRONOUS"]:
                        job_id = enqueue(simulation, _session_id())
                        return jsonify(
                            {
                                "job_id": job_id,
                                "status_url": url_for("run_status", job_id=job_id),
                            }
                        ), 202
                    progress_id = request.form.get("progress_id", "")
                    _set_run_progress(progress_id, "Grabbing and processing input data")
                    with _SURF_RUN_LOCK:
                        context["result"] = run_generated_code(
                            context["code"],
                            before_solve=lambda: _set_run_progress(
                                progress_id, "Running SURF"
                            ),
                            on_chunk=lambda current, total: _set_run_progress(
                                progress_id,
                                f"Running SURF — chunk {current} of {total}",
                            ),
                        )
                    if context["result"].success and context["result"].model is not None:
                        context["default_plot_bodies"] = _default_plot_bodies(
                            context["result"].model
                        )
                        context["plot_body_choices"] = _available_plot_body_choices(
                            context["result"].model
                        )
                        context["default_insitu_source"] = _default_insitu_source(
                            context["result"].model.time_init
                        )
                        context["run_id"] = _retain_model(
                            context["result"].model,
                            simulation,
                            getattr(context["result"], "ambient_model", None),
                        )
                        context["ambient_model_available"] = (
                            getattr(context["result"], "ambient_model", None) is not None
                        )
                        context["show_movies"] = not bool(
                            simulation.model.get("is_1d", False)
                        )
            except (
                DonkiAccessError,
                json.JSONDecodeError,
                TypeError,
                ValueError,
            ) as exc:
                if (
                    request.form.get("action") == "run"
                    and not app.config["RUN_JOBS_SYNCHRONOUS"]
                ):
                    return jsonify({"error": str(exc)}), 400
                context["error"] = str(exc)
        return render_template("index.html", **context, **_model_defaults())

    @app.get("/runs/<job_id>/status")
    def run_status(job_id: str):
        """Return persistent background-job state to its submitting browser."""
        status = read_status(job_id)
        if status is None or status.get("owner_session_id") != _session_id():
            abort(404)
        response = {
            key: status.get(key)
            for key in ("id", "state", "message", "created_at", "updated_at")
        }
        if status.get("state") in {"completed", "failed"}:
            response["result_url"] = url_for("index", run_id=job_id)
        return jsonify(response)

    @app.get("/run-progress/<progress_id>")
    def run_progress(progress_id: str):
        """Return the latest processing phase for an in-flight run."""
        progress_key = f"{_session_id()}:{progress_id}"
        with _RUN_PROGRESS_LOCK:
            message = _RUN_PROGRESS.get(progress_key, "")
        return jsonify({"message": message})

    @app.get("/runs/<run_id>/plot/<kind>.png")
    def plot(run_id: str, kind: str):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import astropy.units as u
        import surf.surf_analysis as sa

        retained = _run_for(run_id)
        model = retained["model"]
        ambient_model = retained.get("ambient_model")
        run_name, ambient_name = _run_names(retained)
        show_ambient = (
            request.args.get("show_ambient") == "1" and ambient_model is not None
        )
        with _PLOT_LOCK:
            plt.close("all")
            if kind == "map":
                plot_time = float(request.args.get("time", 1.5)) * u.day
                simulation = retained.get("simulation")
                solver = (
                    str(simulation.model["solver"]).lower()
                    if isinstance(simulation, SimulationRequest)
                    else str(getattr(model, "solver", "huxt")).lower()
                )
                options = {
                    "minimalplot": request.args.get("minimal") == "1",
                    "plotHCS": request.args.get("plot_hcs", "1") == "1",
                    "annotateplot": request.args.get("annotate", "1") == "1",
                    "show_body_latitudes": (
                        request.args.get("show_body_latitudes") == "1"
                    ),
                    "bodies": _requested_plot_bodies(),
                    "plot_rmax": (
                        float(request.args["plot_rmax"])
                        if request.args.get("plot_rmax")
                        else None
                    ),
                }
                if "huxt" in solver:
                    options["trace_earth_connection"] = (
                        request.args.get("trace_earth") == "1"
                    )
                    with _horizons_plot_observers(model, sa, options["bodies"]):
                        sa.plot(model, plot_time, **options)
                else:
                    with _horizons_plot_observers(model, sa, options["bodies"]):
                        sa.plot_compressible(model, plot_time, **options)
            elif kind == "radial":
                _figure, radial_axes = plot_radial_profile(
                    model,
                    float(request.args.get("radial_time", 1.5)) * u.day,
                    lon=float(request.args.get("radial_lon", 0)) * u.deg,
                )
                if show_ambient:
                    import numpy as np

                    axes = np.atleast_1d(plt.gcf().axes)
                    time = float(request.args.get("radial_time", 1.5)) * u.day
                    lon = float(request.args.get("radial_lon", 0)) * u.deg
                    ti = int(np.argmin(np.abs(ambient_model.time_out - time)))
                    li = 0 if ambient_model.lon.size == 1 else int(
                        np.argmin(np.abs(ambient_model.lon - lon))
                    )
                    axes[0].plot(
                        ambient_model.r,
                        ambient_model.v_grid[ti, :, li],
                        "b--", label="Ambient (no CMEs)",
                    )
                    if getattr(ambient_model, "compressible", False):
                        axes[1].semilogy(
                            ambient_model.r,
                            ambient_model.rho_grid[ti, :, li].value / 1.6726e-27 / 1e6,
                            "b--", label="Ambient (no CMEs)",
                        )
                        axes[2].semilogy(
                            ambient_model.r, ambient_model.temp_grid[ti, :, li].value,
                            "b--", label="Ambient (no CMEs)",
                        )
                    for axis in axes:
                        axis.legend()
            elif kind == "timeseries":
                observer = request.args.get("observer", "custom")
                if observer == "custom":
                    plot_custom_timeseries(
                        model,
                        float(request.args.get("radius", 1)) * u.AU,
                        lon=float(request.args.get("timeseries_lon", 0)) * u.deg,
                    )
                elif observer == "Earth":
                    plot_omni = request.args.get(
                        "plot_insitu", request.args.get("plot_omni", "1")
                    ) == "1"
                    insitu_source = request.args.get("insitu_source", "OMNI").upper()
                    if insitu_source not in {"OMNI", "SWPC"}:
                        abort(400, "Unknown Earth observation source.")
                    try:
                        figure, axes = sa.plot_earth_timeseries(
                            model, plot_omni=plot_omni, insitu_source=insitu_source
                        )
                    except Exception:
                        if not plot_omni:
                            raise
                        app.logger.warning(
                            "%s data could not be plotted; returning the SURF-only "
                            "Earth time series.",
                            insitu_source,
                            exc_info=True,
                        )
                        plt.close("all")
                        figure, axes = sa.plot_earth_timeseries(
                            model, plot_omni=False
                        )
                    earth_series = sa.get_observer_timeseries(model, observer="Earth")
                    format_datetime_axis_like_surf(
                        figure,
                        axes,
                        earth_series["time"],
                    )
                else:
                    import numpy as np

                    target_key = observer.upper().replace(" ", "_")
                    series = (
                        _horizons_timeseries(model, target_key)
                        if target_key in _HORIZONS_TARGETS
                        else sa.get_observer_timeseries(model, observer=observer)
                    )
                    fields = [
                        (key, label)
                        for key, label in (
                            ("vsw", "V [km/s]"),
                            ("bpol", r"B$_{\mathrm{POL}}$"),
                            ("n", r"n$_\mathrm{P}$ [cm$^{-3}$]"),
                            ("T", "T [K]"),
                        )
                        if key in series
                        and np.isfinite(np.asarray(series[key], dtype=float)).any()
                    ]
                    figure, axes = plt.subplots(
                        len(fields), 1, figsize=(14, 3 * len(fields)), sharex=True
                    )
                    for axis, (key, label) in zip(np.atleast_1d(axes), fields):
                        if key in {"n", "T"}:
                            axis.semilogy(
                                series["time"], series[key], "r-", label="SURF"
                            )
                        elif key == "bpol":
                            axis.plot(
                                series["time"], np.sign(series[key]), "r.", label="SURF"
                            )
                        else:
                            axis.plot(series["time"], series[key], "r-", label="SURF")
                        axis.set_ylabel(label)
                        if key == "vsw":
                            axis.set_ylim(250, 1000)
                        elif key == "bpol":
                            axis.set_ylim(-1.1, 1.1)
                        elif key == "n":
                            axis.set_ylim(0.101, 999)
                        elif key == "T":
                            axis.set_ylim(1e4, 9.9e6)
                        axis.grid(True, alpha=0.3)
                        axis.legend()
                    observation_sources = {
                        "PSP": ("get_psp", "Parker Solar Probe"),
                        "SOLO": ("get_solo", "Solar Orbiter"),
                        "STA": ("get_stereo_a", "STEREO-A"),
                    }
                    observation_source = observation_sources.get(observer)
                    plot_insitu = request.args.get("plot_insitu", "1") == "1"
                    if observation_source and plot_insitu:
                        try:
                            import surf.surf_insitu as sinsit

                            grabber_name, observation_label = observation_source
                            observations = getattr(sinsit, grabber_name)(
                                series["time"].iloc[0], series["time"].iloc[-1]
                            )
                            observation_fields = {
                                "vsw": "V",
                                "bpol": "BR",
                                "n": "N",
                                "T": "T",
                            }
                            for axis, (key, _label) in zip(np.atleast_1d(axes), fields):
                                observation_key = observation_fields.get(key)
                                if observation_key not in observations:
                                    continue
                                values = observations[observation_key]
                                if key == "bpol":
                                    values = np.sign(values) * 0.92
                                    axis.plot(
                                        observations["datetime"],
                                        values,
                                        "k.",
                                        label=observation_label,
                                    )
                                elif key in {"n", "T"}:
                                    axis.semilogy(
                                        observations["datetime"],
                                        values,
                                        "k-",
                                        label=observation_label,
                                    )
                                else:
                                    axis.plot(
                                        observations["datetime"],
                                        values,
                                        "k-",
                                        label=observation_label,
                                    )
                                axis.legend()
                        except Exception:
                            app.logger.warning(
                                "%s data could not be plotted; returning the "
                                "SURF-only time series.",
                                observation_source[1],
                                exc_info=True,
                            )
                    format_datetime_axis_like_surf(
                        figure,
                        np.atleast_1d(axes),
                        series["time"],
                    )
                    figure.subplots_adjust(
                        left=0.10,
                        bottom=0.14,
                        right=0.98,
                        top=0.90,
                        hspace=0.05,
                    )
                    figure.suptitle(f"SURF time series at {observer}")
                if show_ambient:
                    import numpy as np
                    from surfs_up.core.plotting import sample_custom_timeseries

                    ambient_series = (
                        sample_custom_timeseries(
                            ambient_model,
                            float(request.args.get("radius", 1)) * u.AU,
                            float(request.args.get("timeseries_lon", 0)) * u.deg,
                        )
                        if observer == "custom"
                        else (
                            _horizons_timeseries(ambient_model, target_key)
                            if observer != "Earth" and target_key in _HORIZONS_TARGETS
                            else sa.get_observer_timeseries(
                                ambient_model, observer=observer
                            )
                        )
                    )
                    ambient_times = ambient_series.get(
                        "time", ambient_series.get("time_days")
                    )
                    plot_axes = np.atleast_1d(plt.gcf().axes)
                    keys = [
                        key for key in ("vsw", "bpol", "n", "T")
                        if key in ambient_series
                        and np.isfinite(np.asarray(ambient_series[key], dtype=float)).any()
                    ]
                    for axis, key in zip(plot_axes, keys):
                        values = ambient_series[key]
                        if key == "bpol":
                            values = np.sign(values)
                        axis.plot(
                            ambient_times, values, "b--", label="Ambient (no CMEs)"
                        )
                        axis.legend()
            else:
                abort(404)
            if kind == "timeseries":
                style_timeseries_legends(
                    plt.gcf().axes,
                    run_name,
                    ambient_name if show_ambient else None,
                )
            elif kind == "map":
                for artist in plt.gcf().findobj(match=lambda item: hasattr(item, "get_text")):
                    artist.set_text(
                        _replace_run_text(artist.get_text(), run_name, ambient_name)
                    )
            output = io.BytesIO()
            plt.gcf().savefig(output, format="png", dpi=140, bbox_inches="tight")
            plt.close("all")
        output.seek(0)
        return send_file(output, mimetype="image/png")

    @app.post("/ambient-plot.png")
    def ambient_plot():
        import matplotlib.pyplot as plt

        with _PLOT_LOCK:
            figure = _ambient_preview_figure()
            output = io.BytesIO()
            figure.savefig(output, format="png", dpi=140, bbox_inches="tight")
            plt.close("all")
        output.seek(0)
        return send_file(output, mimetype="image/png")

    @app.get("/donki-cmes")
    def donki_cmes():
        start = datetime.datetime.fromisoformat(request.args["start"].replace("T", " "))
        duration = float(request.args.get("duration", 10))
        try:
            return jsonify(
                _fetch_donki_cmes(
                    start,
                    duration,
                    request.args.get("solver", "huxt"),
                    request.args.get("feature", "LE"),
                )
            )
        except DonkiAccessError as exc:
            abort(502, str(exc))

    @app.get("/runs/<run_id>/timeseries.csv")
    def timeseries_csv(run_id: str):
        import astropy.units as u
        import pandas as pd
        import surf.surf_analysis as sa

        retained = _run_for(run_id)
        model = retained["model"]
        observer = request.args.get("observer", "Earth")
        if observer == "custom":
            radius = float(request.args.get("radius", 1)) * u.AU
            longitude = float(request.args.get("timeseries_lon", 0)) * u.deg
            series = sample_custom_timeseries(model, radius, longitude)
        else:
            target_key = observer.upper().replace(" ", "_")
            series = (
                _horizons_timeseries(model, target_key)
                if target_key in _HORIZONS_TARGETS
                else sa.get_observer_timeseries(model, observer=observer)
            )

        if hasattr(series, "copy") and hasattr(series, "columns"):
            surf_frame = series.copy()
        else:
            normalized = {}
            for key, values in series.items():
                if hasattr(values, "value"):
                    values = values.value
                normalized[key] = values
            surf_frame = pd.DataFrame(normalized)
        if surf_frame.empty:
            abort(500, "The SURF time series contains no values.")

        time_column = "time" if "time" in surf_frame else "time_days"
        surf_frame = surf_frame.rename(
            columns={
                column: f"SURF_{column}"
                for column in surf_frame.columns
                if column != time_column
            }
        ).rename(columns={time_column: "time"})
        if time_column == "time":
            surf_frame["time"] = pd.to_datetime(surf_frame["time"])
        else:
            surf_frame["time"] = (
                pd.Timestamp(model.time_init)
                + pd.to_timedelta(surf_frame["time"], unit="D")
            )

        observation_sources = {
            "Earth": ("get_omni", "OMNI"),
            "PSP": ("get_psp", "PSP"),
            "SOLO": ("get_solo", "SOLO"),
            "STA": ("get_stereo_a", "STA"),
        }
        observation_source = observation_sources.get(observer)
        plot_insitu = request.args.get(
            "plot_insitu", request.args.get("plot_omni", "1")
        ) == "1"
        if observation_source and plot_insitu and time_column == "time":
            try:
                import surf.surf_insitu as sinsit

                grabber_name, prefix = observation_source
                observations = getattr(sinsit, grabber_name)(
                    surf_frame["time"].iloc[0], surf_frame["time"].iloc[-1]
                )
                exported_fields = [
                    field
                    for field in ("datetime", "V", "N", "T", "BR", "BX_GSE", "B")
                    if field in observations
                ]
                observation_frame = observations[exported_fields].copy()
                observation_frame = observation_frame.rename(
                    columns={
                        field: "time" if field == "datetime" else f"{prefix}_{field}"
                        for field in exported_fields
                    }
                )
                observation_frame["time"] = pd.to_datetime(observation_frame["time"])
                surf_frame = pd.merge(
                    surf_frame,
                    observation_frame,
                    on="time",
                    how="outer",
                    sort=True,
                )
            except Exception:
                app.logger.warning(
                    "%s data could not be exported; writing SURF values only.",
                    observation_source[1],
                    exc_info=True,
                )

        # Serialize timestamps ourselves because pandas' CSV default uses a
        # space separator rather than the ISO 8601 ``T`` separator.
        surf_frame["time"] = surf_frame["time"].map(
            lambda value: "" if pd.isna(value) else pd.Timestamp(value).isoformat()
        )
        csv_text = surf_frame.to_csv(index=False)
        payload = io.BytesIO(csv_text.encode("utf-8"))
        return send_file(
            payload,
            mimetype="text/csv",
            as_attachment=True,
            download_name=f"SURF_{observer}_timeseries.csv",
        )

    @app.get("/runs/<run_id>/movie/<kind>.mp4")
    def movie(run_id: str, kind: str):
        _configure_animation_ffmpeg()
        import surf.surf_analysis as sa

        retained = _run_for(run_id)
        model = retained["model"]
        ambient_model = retained.get("ambient_model")
        run_name, ambient_name = _run_names(retained)
        duration = float(request.args.get("duration", 10))
        fps = int(request.args.get("fps", 5))
        with tempfile.TemporaryDirectory() as directory, _PLOT_LOCK:
            path = Path(directory) / "surf_movie.mp4"
            options = {
                "tag": request.args.get("tag", "gui"),
                "duration": duration,
                "fps": fps,
                "plotHCS": request.args.get("plot_hcs", "1") == "1",
                "show_body_latitudes": (
                    request.args.get("show_body_latitudes") == "1"
                ),
                "bodies": _requested_plot_bodies(),
                "plot_rmax": (
                    float(request.args["plot_rmax"])
                    if request.args.get("plot_rmax")
                    else None
                ),
                "outputfilepath": str(path),
            }
            if kind == "map":
                options["trace_earth_connection"] = request.args.get("trace_earth") == "1"
                animation = sa.animate
            elif kind == "timeseries":
                options["polar_var"] = request.args.get("field", "V")
                options["plot_omni"] = request.args.get("plot_omni", "1") == "1"
                options["insitu_source"] = request.args.get(
                    "insitu_source", "OMNI"
                ).upper()
                if (
                    request.args.get("show_ambient") == "1"
                    and ambient_model is not None
                ):
                    options["model_ambient"] = ambient_model
                animation = sa.animate_with_ts
            else:
                abort(404)
            signature = inspect.signature(animation)
            accepts_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in signature.parameters.values()
            )
            supported_options = (
                options
                if accepts_kwargs
                else {
                    key: value
                    for key, value in options.items()
                    if key in signature.parameters
                }
            )
            original_plot_compressible = None
            if animation is sa.animate and getattr(model, "compressible", False):
                original_plot_compressible = sa.plot_compressible
                sa.plot_compressible = _hydro_plot_with_calendar_date(
                    original_plot_compressible
                )
            try:
                with _movie_run_labels(run_name, ambient_name):
                    with _horizons_plot_observers(model, sa, options["bodies"]):
                        saved = animation(model, **supported_options)
            finally:
                if original_plot_compressible is not None:
                    sa.plot_compressible = original_plot_compressible
            movie_path = Path(saved) if saved else path
            payload = io.BytesIO(movie_path.read_bytes())
            movie_is_gif = movie_path.suffix.lower() == ".gif"
        payload.seek(0)
        return send_file(
            payload,
            mimetype="image/gif" if movie_is_gif else "video/mp4",
            as_attachment=request.args.get("inline") != "1",
            download_name=f"SURF_{run_id[:8]}.{'gif' if movie_is_gif else 'mp4'}",
        )

    @app.get("/runs/<run_id>/movie/<kind>.gif")
    def legacy_kind_movie(run_id: str, kind: str):
        """Redirect old per-kind GIF URLs to the default MP4 renderer."""
        return redirect(url_for("movie", run_id=run_id, kind=kind, **request.args))

    @app.get("/runs/<run_id>/movie.gif")
    def legacy_movie(run_id: str):
        """Keep old bookmarked movie URLs working."""
        return redirect(url_for("movie", run_id=run_id, kind="map", **request.args))

    return app


def main() -> None:
    """Run the development server, worker, and open the interface."""
    from surfs_up.jobs import cancel_all_unfinished_jobs, run_worker

    # A prior local process cannot still be doing useful work after it has
    # exited. Do not replay its abandoned queue before the user presses Run.
    cancel_all_unfinished_jobs()
    threading.Thread(
        target=run_worker,
        daemon=True,
        name="surfs-up-worker",
    ).start()
    # Give Flask a moment to bind its socket before asking the default browser
    # to load the interface. The timer is a daemon so it cannot delay shutdown.
    browser_timer = threading.Timer(
        1.0,
        webbrowser.open,
        args=("http://127.0.0.1:5000",),
    )
    browser_timer.daemon = True
    browser_timer.start()
    # Disabling Werkzeug's process reloader prevents it from starting a second
    # worker and racing for the same filesystem queue.
    create_app().run(debug=True, use_reloader=False)


if __name__ == "__main__":
    main()
