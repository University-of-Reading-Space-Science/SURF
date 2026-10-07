import datetime
import io
from pathlib import Path
from urllib.error import URLError

import pytest

from surfs_up.core import SimulationRequest
from surfs_up.core.codegen import build_generated_code
from surfs_up.web import app


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


def test_browser_opens_only_after_server_is_ready(monkeypatch):
    events = []
    attempts = iter([URLError("not ready"), _Response()])

    def open_url(url, timeout):
        events.append(("probe", url, timeout))
        result = next(attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(app, "urlopen", open_url)
    monkeypatch.setattr(app.time, "sleep", lambda delay: events.append(("sleep", delay)))
    monkeypatch.setattr(app.webbrowser, "open", lambda url: events.append(("open", url)))

    assert app._open_browser_when_ready("http://127.0.0.1:5000", retry_delay=0.01)
    assert events == [
        ("probe", "http://127.0.0.1:5000/healthz", 0.5),
        ("sleep", 0.01),
        ("probe", "http://127.0.0.1:5000/healthz", 0.5),
        ("open", "http://127.0.0.1:5000"),
    ]


def test_generated_plot_code_includes_selected_ambient_solution():
    template = (
        Path(app.__file__).with_name("templates") / "index.html"
    ).read_text(encoding="utf-8")

    assert 'document.getElementById("radial-show-ambient")?.checked' in template
    assert 'document.getElementById("timeseries-show-ambient")?.checked' in template
    assert "ambient_model.v_grid[ti, :, li]" in template
    assert "sa.get_observer_timeseries(ambient_model" in template
    assert "sample_custom_timeseries(ambient_model" in template


def test_surfs_up_defaults_to_hydro_insitu_and_ambient_comparison():
    template = (
        Path(app.__file__).with_name("templates") / "index.html"
    ).read_text(encoding="utf-8")

    assert '<option value="hydro" selected>hydro</option>' in template
    assert 'id="ambient-source" value="insitu_backmapped"' in template
    assert 'name="run_ambient_solution" type="checkbox" checked' in template
    assert 'id="radial-show-ambient" class="show-ambient-solution" type="checkbox" checked' in template
    assert 'id="timeseries-show-ambient" class="show-ambient-solution" type="checkbox" checked' in template
    assert 'id="movie-ts-ambient" type="checkbox" checked' in template
    assert 'showAmbientPanel("insitu_backmapped")' in template


def test_completed_run_code_is_not_regenerated_but_accepts_plot_history():
    template = (
        Path(app.__file__).with_name("templates") / "index.html"
    ).read_text(encoding="utf-8")

    assert "let codeIsExecuted = {{ code_is_executed|tojson }};" in template
    assert "if (plotCodeHistory.length)" in template
    assert "if (!codeIsExecuted && plotCodeHistory.length)" not in template
    assert "function recordPlotCode(codeText) {\n      plotCodeHistory.push" in template
    assert "if (codeIsExecuted) return;" in template
    assert "if (codeIsExecuted) {" in template
    assert "Executed Code" in template


def test_changing_a_completed_run_restores_live_code_preview():
    template = (
        Path(app.__file__).with_name("templates") / "index.html"
    ).read_text(encoding="utf-8")

    invalidation = template.split("function invalidateCompletedRun()", 1)[1].split(
        "runForm.addEventListener", 1
    )[0]
    assert "runIsCurrent = false;" in invalidation
    assert "codeIsExecuted = false;" in invalidation
    assert invalidation.index("codeIsExecuted = false;") < invalidation.index(
        "resetGeneratedCodeForCurrentState();"
    )


def test_insitu_forecast_and_model_start_are_only_linked_for_defaults():
    template = (
        Path(app.__file__).with_name("templates") / "index.html"
    ).read_text(encoding="utf-8")

    assert "if (!restoredPreRunConfiguration) {\n      syncOmniForecastTime();" in template
    assert "syncModelStartFromOmniForecastTime" not in template

    start_change_handler = template.split(
        'modelStartInput.addEventListener("change", () => {', 1
    )[1].split("});", 1)[0]
    start_input_handler = template.split(
        'modelStartInput.addEventListener("input", () => {', 1
    )[1].split("});", 1)[0]
    assert "syncOmniForecastTime" not in start_change_handler
    assert "syncOmniForecastTime" not in start_input_handler


@pytest.mark.parametrize("payload", [b"", b"<html>Internal error</html>", b"{}"])
def test_donki_invalid_responses_become_clean_access_errors(monkeypatch, payload):
    monkeypatch.setattr(app, "urlopen", lambda *_args, **_kwargs: io.BytesIO(payload))

    with pytest.raises(app.DonkiAccessError, match="temporarily unavailable"):
        app._fetch_donki_cmes(datetime.datetime(2026, 10, 1), 5, "hydro", "LE")


def test_grab_donki_now_uses_the_same_endpoint_as_surf_runs():
    assert app._DONKI_URL == (
        "https://ccmc.gsfc.nasa.gov/DONKI-API/get/CMEAnalysis"
    )


def test_insitu_forecast_uses_gui_start_to_compute_buffer():
    simulation = SimulationRequest.from_mappings(
        {
            "solver": "hydro",
            "start_datetime": "2026-09-28T00:00:00",
            "simtime_days": 15,
            "rmin": 21.5,
            "rmax": 240,
            "latitude": 0,
            "cr_num": 2300,
            "cr_lon_init_deg": 0,
        },
        {
            "source": "insitu_backmapped",
            "mode": "forecast",
            "spacecraft": "OMNI",
            "forecast_datetime": "2026-10-07T00:00:00",
            "icme_list": "None",
        },
        [],
    )

    code = build_generated_code(simulation)

    assert "forecast_time = datetime.datetime.fromisoformat('2026-10-07T00:00:00')" in code
    assert "forecast_buffer = ((forecast_time - start_time).total_seconds()/86400)*u.day" in code
    assert "buffertime=forecast_buffer" in code
    assert "buffertime=5*u.day" not in code


def test_run_start_donki_query_uses_constructed_model_interval():
    simulation = SimulationRequest.from_mappings(
        {
            "solver": "hydro",
            "start_datetime": "2026-09-28T00:00:00",
            "simtime_days": 15,
            "rmin": 21.5,
            "rmax": 240,
            "latitude": 0,
            "cr_num": 2300,
            "cr_lon_init_deg": 0,
            "grab_donki_at_run_start": True,
        },
        {
            "source": "insitu_backmapped",
            "mode": "forecast",
            "spacecraft": "OMNI",
            "forecast_datetime": "2026-10-07T00:00:00",
            "icme_list": "None",
        },
        [],
    )

    code = build_generated_code(simulation)

    assert "donki_start_time = model.time_init.to_datetime()" in code
    assert "donki_end_time = donki_start_time + datetime.timedelta" in code
    assert "get_DONKI_cme_list(model, donki_start_time, donki_end_time" in code
    assert "Querying DONKI cone CMEs from" in code
