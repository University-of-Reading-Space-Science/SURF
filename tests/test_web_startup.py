from urllib.error import URLError

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
