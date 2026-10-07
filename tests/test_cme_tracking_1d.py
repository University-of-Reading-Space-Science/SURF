import numpy as np
from astropy import units as u

import surf.surf as s


def test_chunked_1d_hydro_materializes_cme_coordinates_once(monkeypatch):
    calls = []
    original_track = s.ConeCME._track_

    def counted_track(cme, model, cme_id):
        calls.append((cme, cme_id))
        return original_track(cme, model, cme_id)

    monkeypatch.setattr(s.ConeCME, "_track_", counted_track)
    model = s.SURF(
        v_boundary=np.full(128, 400.0) * u.km / u.s,
        simtime=0.5 * u.day,
        solver="hydro",
        track_cmes=True,
        lon_out=0 * u.deg,
    )
    cme = s.ConeCME(
        t_launch=0.1 * u.day,
        longitude=0 * u.deg,
        latitude=0 * u.deg,
        width=60 * u.deg,
        v=900 * u.km / u.s,
        thickness=2 * u.solRad,
    )

    s.solve_chunked(
        model,
        [cme],
        chunk_simtime=0.25 * u.day,
        verbose=False,
    )

    assert len(calls) == 1
    tracked_cme = model.cmes[0]
    assert len(tracked_cme.coords) == model.nt_out
    assert any(entry["r"].size for entry in tracked_cme.coords.values())
