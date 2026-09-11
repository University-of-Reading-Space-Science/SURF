"""Session-lifetime cache for SURF in-situ data downloads."""

from __future__ import annotations

import datetime
import functools
import threading
from dataclasses import dataclass

import pandas as pd

_PADDING = datetime.timedelta(days=27)
_DOWNLOADERS = (
    "get_omni", "get_SWPC_realtime", "get_stereo_a", "get_psp", "get_solo"
)
_LOCK = threading.RLock()
_ORIGINALS: dict[str, object] = {}


@dataclass
class _Entry:
    start: pd.Timestamp
    end: pd.Timestamp
    frame: pd.DataFrame


_CACHE: dict[tuple[object, ...], _Entry] = {}


class InsituDataUnavailableError(RuntimeError):
    """Raised when an upstream archive returns no files for a request."""


def _timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _slice(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if "datetime" not in frame:
        return frame.copy(deep=True)
    times = pd.to_datetime(frame["datetime"])
    if getattr(times.dt, "tz", None) is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    return frame.loc[(times >= start) & (times <= end)].copy(deep=True).reset_index(drop=True)


def _raise_clear_empty_download_error(
    name: str, start: pd.Timestamp, end: pd.Timestamp, error: Exception
) -> None:
    """Translate SunPy's empty-TimeSeries implementation error at our boundary."""
    if not (isinstance(error, IndexError) and str(error) == "pop from empty list"):
        raise error

    source = {
        "get_stereo_a": "STEREO-A/CDAWeb",
        "get_omni": "OMNI",
        "get_SWPC_realtime": "SWPC real-time",
        "get_psp": "Parker Solar Probe",
        "get_solo": "Solar Orbiter",
    }.get(name, name)
    interval = f"{start.isoformat()} to {end.isoformat()}"
    raise InsituDataUnavailableError(
        f"No {source} files are available for {interval}. "
        "The selected interval may be outside the archive's coverage or newer "
        "than its latest published data; choose an earlier forecast time or a "
        "different in-situ source."
    ) from error


def _cached_call(name: str, original, starttime, endtime, *args, **kwargs):
    start = _timestamp(starttime)
    end = _timestamp(endtime)
    if end < start:
        raise ValueError("endtime must be on or after starttime")
    key = (name, args, tuple(sorted(kwargs.items())))
    with _LOCK:
        entry = _CACHE.get(key)
        if entry is not None and entry.start <= start and entry.end >= end:
            return _slice(entry.frame, start, end)

        fetch_start = min(start, entry.start) - _PADDING if entry else start - _PADDING
        fetch_end = max(end, entry.end) + _PADDING if entry else end + _PADDING
        try:
            frame = original(fetch_start.to_pydatetime(), fetch_end.to_pydatetime(), *args, **kwargs)
            coverage_start, coverage_end = fetch_start, fetch_end
        except Exception:
            # Spacecraft and real-time archives can have hard coverage limits.
            # Preserve their normal behavior while still caching the exact request.
            try:
                frame = original(starttime, endtime, *args, **kwargs)
            except Exception as error:
                _raise_clear_empty_download_error(name, start, end, error)
            coverage_start, coverage_end = start, end
        if not isinstance(frame, pd.DataFrame):
            return frame
        _CACHE[key] = _Entry(coverage_start, coverage_end, frame.copy(deep=True))
        return _slice(frame, start, end)


def install_insitu_download_cache() -> None:
    """Clear the cache and wrap SURF downloaders for this application process."""
    import surf.surf_insitu as sinsit

    with _LOCK:
        _CACHE.clear()
        for name in _DOWNLOADERS:
            current = getattr(sinsit, name, None)
            if current is None:
                continue
            if name not in _ORIGINALS:
                _ORIGINALS[name] = current
            original = _ORIGINALS[name]

            @functools.wraps(original)
            def cached(starttime, endtime, *args, __name=name, __original=original, **kwargs):
                return _cached_call(
                    __name, __original, starttime, endtime, *args, **kwargs
                )

            setattr(sinsit, name, cached)


def clear_insitu_download_cache() -> None:
    """Discard all downloaded in-situ frames."""
    with _LOCK:
        _CACHE.clear()
