from astropy.time import Time, TimeDelta
import astropy.units as u
import numpy as np
from surf import surf as s

def test_synthetic_observer():
    """
    Function to test if the synthetic observer option of the Observer class functions as expected.
    Test this by constructing a synthetic observer inbodynce from the Observer class of a
    specified body. If working, the synthetic inbodynce and real inbodynce should match.
    Returns:
    """
    # Get a sample of times - 6 hourly for a year.
    time = Time('2025-01-01T00:00:00')
    times = time + TimeDelta(np.arange(0, 365, 0.25), format='jd')

    craft = ["ACE", "STA", "PSP", "SOLO"]
    planets = ["MERCURY", "EARTH", "VENUS", "MARS", "JUPITER", "SATURN"]
    bodies = planets + craft
    for b in bodies:
        body = s.Observer(body=b, times=times)

        # Get a sample of positions, and skip some to test the interpolation.
        synth_heeq_pos = {'time': times[::1],
                    'r': body.r[::1],
                    'lon': body.lon[::1],
                    'lat': body.lat[::1]}

        synth = s.Observer(body='SYNTHETIC', times=times, synth_heeq_pos=synth_heeq_pos)

        def _wrap_deg(diff):
            """Fold an angular difference into [-180, 180] degrees."""
            d = diff.to(u.deg)
            return (d + 180 * u.deg) % (360 * u.deg) - 180 * u.deg

        # HEEQ should round-trip to (near) machine precision.
        assert np.allclose(body.r.to(u.solRad).value, synth.r.to(u.solRad).value,
                           rtol=1e-5, atol=1e-4)
        assert np.allclose(_wrap_deg(body.lon - synth.lon).value, 0.0, atol=1e-3)
        assert np.allclose(body.lat.to(u.deg).value, synth.lat.to(u.deg).value, atol=1e-3)

        # Carrington and HAE carry small differences because the synthetic
        # observer reconstructs the frame without the Horizons observer
        # geometry (light-travel / aberration terms). Use looser tolerances.
        assert np.allclose(_wrap_deg(body.lon_c - synth.lon_c).value, 0.0, atol=0.5)
        assert np.allclose(_wrap_deg(body.lon_hae - synth.lon_hae).value, 0.0, atol=0.5)
