import datetime
import json

import pytest
import pandas as pd
from astropy.time import Time

import surf.surf_insitu as sinsit


class _Response:
    status = 200

    def __init__(self, events):
        self._events = events

    def read(self):
        return json.dumps(self._events).encode("utf-8")


def test_donki_icmes_default_to_quality_one_or_better(monkeypatch):
    events = [
        {
            "location": "Earth",
            "eventTime": f"2026-01-0{quality + 3}T00:00Z",
            "quality": quality,
        }
        for quality in (-1, 0, 1, 2)
    ]
    monkeypatch.setattr(sinsit, "urlopen", lambda url: _Response(events))

    result = sinsit.get_DONKI_ICMEs(
        datetime.datetime(2026, 1, 1), datetime.datetime(2026, 1, 10)
    )

    assert result["quality"].tolist() == [1, 2]


def test_donki_icme_quality_threshold_can_include_unspecified(monkeypatch):
    events = [{
        "location": "Earth",
        "eventTime": "2026-01-01T00:00Z",
        "quality": None,
    }]
    monkeypatch.setattr(sinsit, "urlopen", lambda url: _Response(events))

    result = sinsit.get_DONKI_ICMEs(
        datetime.datetime(2026, 1, 1),
        datetime.datetime(2026, 1, 2),
        min_quality=-1,
    )

    assert len(result) == 1


def test_donki_icmes_return_empty_normalized_schema_when_none_meet_quality(monkeypatch):
    events = [{
        "location": "Earth",
        "eventTime": "2026-01-01T00:00Z",
        "quality": 0,
    }]
    monkeypatch.setattr(sinsit, "urlopen", lambda url: _Response(events))

    result = sinsit.get_DONKI_ICMEs(
        datetime.datetime(2026, 1, 1),
        datetime.datetime(2026, 1, 2),
        min_quality=1,
    )

    assert result.empty
    assert "Shock_time" in result
    assert "ICME_end" in result

    monkeypatch.setattr(sinsit, "get_DONKI_ICMEs", lambda *args, **kwargs: result)
    timestamps = pd.date_range("2026-01-01", periods=3, freq="h").to_pydatetime()
    omni = pd.DataFrame({
        "datetime": timestamps,
        "mjd": Time(timestamps).mjd,
        "V": [400.0, 410.0, 420.0],
        "BX_GSE": [1.0, 2.0, 3.0],
    })
    filtered = sinsit.removeICMEs(omni, icme_list="DONKI")
    pd.testing.assert_frame_equal(filtered, omni)


def test_donki_icme_quality_threshold_is_validated():
    with pytest.raises(ValueError, match="min_quality"):
        sinsit.get_DONKI_ICMEs(
            datetime.datetime(2026, 1, 1),
            datetime.datetime(2026, 1, 2),
            min_quality=3,
        )
