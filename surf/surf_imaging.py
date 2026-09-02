"""
This module contains the SyntheticImager class, which generates synthetic heliospheric images from
SURF model output.
"""
import copy
from datetime import datetime
import os

import astropy.constants as const
from astropy.coordinates import SkyCoord, cartesian_to_spherical, spherical_to_cartesian
from sunpy.coordinates import frames
from astropy.time import Time
import astropy.units as u
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid
import skimage as ski

from surf import surf as s
from surf import surf_inputs as sIN
from surf import surf_analysis as sA


class SyntheticImager:
    """
    Class to generate synthetic heliospheric images from SURF model output.
    """

    def __init__(self, observer, pa=90.0, elon_min=5.0, elon_max=30.0):
        """
        Parameters
        ----------
        observer : Observer
            The observer ephemeris object.
        pa : float, optional
            Position angle of the field of view, in degrees. Default is 90.0. Must be in range [
            0,360].
        elon_min : float, optional
            Minimum elongation of the field of view, in degrees. Default is 5.0.
            May be negative to indicate a field of view on the opposite side of the
            observer (i.e. looking in the retrograde direction).
        elon_max : float, optional
            Maximum elongation of the field of view, in degrees. Default is 30.0.
            Must be greater than elon_min, and both values must have the same sign.
        """
        if elon_min <= 0:
            raise ValueError(
                f"elon_min ({elon_min}) must be positive. "
                f"Use elon_sign=-1 to select the retrograde field of view."
            )
        if elon_max <= 0:
            raise ValueError(
                f"elon_max ({elon_max}) must be positive. "
                f"Use elon_sign=-1 to select the retrograde field of view."
            )
        if elon_min >= elon_max:
            raise ValueError(
                f"elon_min ({elon_min}) must be less than elon_max ({elon_max})."
            )

        if (pa <= 0) | (pa >= 360):
            raise ValueError(
                f"pa ({pa}) must be in the range of 0 to 360. "
            )

        self.sigma_e = 7.95 * 10 ** (-30) * u.m ** 2 / u.steradian
        self.u_ld = 0.63  # Limb darkening coefficient. For 5500 angstroms.

        # Store the pa.
        self.pa = pa * u.deg

        # Store full observer ephemeris
        self.position = copy.deepcopy(observer)
        self.position.r = self.position.r.to(u.m)
        self.position.lon = self.position.lon.to(u.rad)
        self.position.lats = self.position.lat.to(u.rad)

        # Set up elongation arrays (time-independent)
        self.e_min, self.e_max, self.e, self.de = self.elon_grid(elon_min, elon_max)

        # Set up LOS distance arrays (time-independent)
        # Note - if you want to look at TS theory, it helps to have Z(elon), so you can define Z
        # to be on the TS and vary as distance away from the TS. Not necessary for this though.
        self.z_min, self.z_max, self.z, self.dz = self.los_distance_grid()

        # Create mesh grid as these don't change with time
        self.e_grid, self.z_grid = np.meshgrid(self.e, self.z)

        # Astropy/Sunpy HPR needs delta not elon.
        self.delta_grid = self.e_grid - 90*u.deg

        self.pa_grid = np.full_like(self.e_grid, self.pa)

        # Initialise FOV geometry using the first time step
        self._compute_fov_geometry(0)


    def _compute_fov_geometry(self, time_step):
        """
        Compute all field-of-view geometry quantities for the observer position
        at the given time step index.

        Updates instance attributes: observer_time, observer_lon, observer_r,
        observer_x, observer_y, omega, e_grid, z_grid, r_grid, chi_grid,
        lon_grid, x_grid, y_grid, g, gr, gt, gp.
        """

        observer_heeq = SkyCoord(
            lon=self.position.lon[time_step],
            lat=self.position.lat[time_step],
            radius=self.position.r[time_step],
            obstime=self.position.time[time_step],
            frame=frames.HeliographicStonyhurst
        )

        los_hpr = SkyCoord(
            psi=self.pa_grid,
            delta=self.delta_grid,
            distance=self.z_grid,
            observer=observer_heeq,
            obstime=observer_heeq.obstime,
            frame=frames.HelioprojectiveRadial
        )

        los_heeq = los_hpr.transform_to(
            frames.HeliographicStonyhurst(obstime=los_hpr.obstime)
            )

        self.r_grid = los_heeq.radius.to(u.m)
        # Normalise longitudes to 0-360, as it plays nicer with SURF
        self.lon_grid = np.rad2deg(np.mod(los_heeq.lon.to_value(u.rad), 2.0 * np.pi)) * u.deg
        self.lat_grid = los_heeq.lat.to(u.deg)

        # Scattering angle of radial illumination at each LOS - using cosine rule
        r_o = self.position.r[time_step]
        r_p = self.r_grid
        r_op = self.z_grid
        self.chi_grid = np.arccos((r_op ** 2 + r_p ** 2 - r_o ** 2) / (2.0 * r_p * r_op))

        # Compute the angular halfwidth of the Sun from the observer's position
        self.omega = self.compute_omega(time_step)

        # Compute the geometric factor for each LOS element
        self.g, self.gr, self.gt, self.gp = self.compute_ts_intensity_factors()

    def elon_grid(self, elon_min=5.0, elon_max=30.0):
        """
        Set up the elongation grid for the synthetic imager.

        The grid is always built from absolute-value elongations so that the
        triangle geometry remains valid. The sign (prograde vs retrograde) is
        handled separately in _compute_fov_geometry via self._elon_sign.
        """

        # Define field of view using absolute elongation values
        elon_min = np.deg2rad(abs(elon_min)) * u.rad
        elon_max = np.deg2rad(abs(elon_max)) * u.rad
        de = np.deg2rad(0.1) * u.rad
        elon = np.arange(elon_min.value, elon_max.value + de.value, de.value) * de.unit

        return elon_min, elon_max, elon, de

    def los_distance_grid(self):
        """
        Set up the line of sight distance grid for the synthetic imager.
        """
        z_min = 0.01 * u.solRad.to(u.m) * u.m
        z_max = 2.0 * u.AU.to(u.m) * u.m
        dz = 0.35 * u.solRad.to(u.m) * u.m
        z = np.arange(z_min.value, z_max.value + dz.value, dz.value) * dz.unit

        return z_min, z_max, z, dz

    def compute_omega(self, time_step):
        """
        Compute the angular halfwidth of the Sun from the observer's position.
        Returns:
            omega: astropy.units.Quantity with units of radians, giving the angular halfwidth of
                   the Sun from the observer's position.
        """

        # Sun center to observer vector:
        omega = np.arcsin(const.R_sun / self.position.r[time_step])
        return omega

    def van_de_hulst_coeffs(self):
        """
        Compute the van de Hulst coefficients.
        Returns:
            tuple: Four van de Hulst coefficients (a, b, c, d)
        """

        so = np.sin(self.omega.to(u.rad).value)
        co = np.cos(self.omega.to(u.rad).value)

        so2 = so * so
        co2 = co * co

        with np.errstate(divide='ignore', invalid='ignore'):
            G = co2 * np.log((1.0 + so) / co) / so
        G = np.nan_to_num(G, nan=0.0, posinf=0.0)

        a = co * so2

        b = -(1.0 - 3.0 * so2 - (1.0 + 3.0 * so2) * G) / 8.0

        c = (4.0 - co * (3.0 + co2)) / 3.0

        d = (5.0 + so2 - (5.0 - so2) * G) / 8.0

        return a, b, c, d

    def compute_ts_intensity_factors(self):
        """
        Compute Thomson Scattering intensity factors for this FoV. Following equations in Xiong et
        al. 2013.
        Returns:
            tuple: Total, Radial and Tangential geometry factors (g, gr, gt)
        """

        scatter_coeff = np.pi * self.sigma_e / 2

        a, b, c, d = self.van_de_hulst_coeffs()
        sc2 = np.sin(self.chi_grid.to(u.rad).value) ** 2

        gt = np.zeros(sc2.shape) + scatter_coeff * ((1 - self.u_ld) * c + self.u_ld * d)  # EQ 1(
        # C) # in Xiong et al. 2013

        gp = scatter_coeff * sc2 * ((1 - self.u_ld) * a + self.u_ld * b)  # EQ 1(D) in Xiong et al.
        # 2013

        g = 2 * gt - gp  # EQ 1 in Xiong et al. 2013

        gr = g - gt  # EQ 1B in Xiong et al. 2013

        return g, gr, gt, gp

    def compute_los_intensity_profile(self, density, elon):
        """
        Compute the total, radial, tangential and polarized intensity for a given density field.
        The density field here is defined along a single line of sight. So has the same
        dimensions as self.z.
        """
        elon_val = elon.to(self.e.unit).value
        diffs = np.abs(self.e.value - elon_val)
        id_e = np.array([np.argmin(diffs)])
        if diffs[id_e[0]] > 0.5 * self.de.value:
            raise ValueError(
                f"Requested elongation {elon} does not match any value in self.e "
                f"(nearest is {self.e[id_e[0]]:.6f}, difference {diffs[id_e[0]]:.2e})"
            )

        I = density * self.g[:, id_e].ravel()  # EQ 5 in Xiong et al. 2013
        Ir = density * self.gr[:, id_e].ravel()
        It = density * self.gt[:, id_e].ravel()
        polarisation = (It - Ir) / (It + Ir)  # EQ 7 in Xiong et al. 2013
        Ip = polarisation * I

        return I, Ir, It, Ip, polarisation

    def compute_all_los_intensity_profile(self, model, time_step):
        """
        Compute the total, radial, tangential and polarized intensity profiles for each LoS.
        """
        # Check that the model is compatible with this FoV.
        self._check_latitude_compatibility(model)

        # Update FOV geometry for the observer position at this time step
        self._compute_fov_geometry(time_step)

        I_fov = np.zeros(self.z_grid.shape) * np.nan
        It_fov = np.zeros(self.z_grid.shape) * np.nan
        Ir_fov = np.zeros(self.z_grid.shape) * np.nan
        Ip_fov = np.zeros(self.z_grid.shape) * np.nan
        pol_fov = np.zeros(self.z_grid.shape) * np.nan

        interpolator = self._density_interpolator(model, time_step)
        fov_mask = self._fov_mask(model)

        # Minimum elongation permitted by the model inner boundary
        elon_min_model = np.arcsin(
            model.SURFlat[0].r[0].to(u.solRad).value /
            self.position.r[time_step].to(u.solRad).value)

        for i in range(I_fov.shape[1]):

            elon = self.e[i]

            # Skip LoS that cross the model inner boundary — leave columns as NaN
            if elon.to(u.rad).value < elon_min_model:
                continue

            mask = fov_mask[:, i]
            n_e_los = self.get_los_density(interpolator, elon, replace_nans=True)
            n_e_los[mask] = 0.0
            I, Ir, It, Ip, polarisation = self.compute_los_intensity_profile(n_e_los, elon)
            I_fov[:, i] = I
            It_fov[:, i] = It
            Ir_fov[:, i] = Ir
            Ip_fov[:, i] = Ip
            pol_fov[:, i] = polarisation

        return I_fov, It_fov, Ir_fov, Ip_fov, pol_fov

    def get_los_density(self, interpolator, elon, replace_nans=True):
        """
        Use the density interpolator to compute the density along a single line of sight.
        """
        id_elon = np.argmin(np.abs(self.e - elon))
        # Get interpolated density along this LOS.
        # SURF longitudes are on [0, 2*pi); normalise transformed HEEQ
        # longitudes before querying the periodic padded axis.
        lon = np.mod(self.lon_grid[:, id_elon].to_value(u.rad), 2.0 * np.pi)
        coords = [self.r_grid[:, id_elon].to_value(u.m).ravel(),
                  lon.ravel(),
                  self.lat_grid[:, id_elon].to_value(u.rad).ravel()]
        coords = np.column_stack(coords)
        n_e_los = interpolator(coords).reshape(self.z.shape)  # * (1.0 / u.m ** 3)

        # Handle any NaNs
        if replace_nans:
            n_e_los = np.nan_to_num(n_e_los, nan=0.0, posinf=0.0, neginf=0.0)

        return n_e_los

    def compute_total_intensity(self, model, time_step):
        """
        Compute the total intensity as a function of elongation, e.g. the total intensity along
        each line of sight.
        """

        I_fov, _, _, _, _ = self.compute_all_los_intensity_profile(model, time_step)
        I_fov = np.nan_to_num(I_fov, nan=0.0, posinf=0.0, neginf=0.0)
        I = trapezoid(I_fov, self.z.value, axis=0)
        return I

    def compute_jmap(self, model):
        """
        Compute the time-elongation intensity map for a given model instance.
        """

        # Check that the model is compatible with this FoV.
        self._check_latitude_compatibility(model)

        time_out = model.SURFlat[0].time_out
        jmap = np.full((self.e.size, time_out.size), np.nan)

        for time_step in range(time_out.size):
            # compute_total_intensity -> compute_all_los_intensity_profile ->
            # _compute_fov_geometry updates geometry for this time step
            I = self.compute_total_intensity(model, time_step)
            jmap[:, time_step] = I

            # Mask elongations that cross the model inner boundary
            elon_min_model = np.arcsin(
                model.SURFlat[0].r[0].to(u.solRad).value /
                self.position.r[time_step].to(u.solRad).value)
            invalid = self.e.to(u.rad).value < elon_min_model
            jmap[invalid, time_step] = np.nan

        djmap = np.diff(jmap, axis=1, prepend=jmap[:, :1])

        return jmap, djmap

    def plot_jmap(self, model, jmap, djmap):
        """
        Make a 2-panel plot of the normal and difference image jmaps
        """

        times = model.SURFlat[0].time_out.to(u.day).value
        elons = self.e.to(u.deg).value

        cmap = plt.cm.gray.copy()
        cmap.set_bad(color='midnightblue')

        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        vmin, vmax = np.nanpercentile(jmap, [1, 99])
        ax[0].pcolormesh(times, elons, jmap, cmap=cmap, vmin=vmin, vmax=vmax)

        vmin, vmax = np.nanpercentile(djmap, [1, 99])
        ax[1].pcolormesh(times, elons, djmap, cmap=cmap, vmin=vmin, vmax=vmax)

        for a in ax:
            a.set_ylim(self.e_min.to(u.deg).value, self.e_max.to(u.deg).value)
            a.set_xlim(times[0], times[-1])
            a.set_xlabel('Time [days]')
            a.set_ylabel('Elongation [deg]')

        fig.subplots_adjust(left=0.05, bottom=0.08, right=0.98, top=0.98, wspace=0.1)
        return fig, ax

    def plot_jmap_with_cme_profiles(self, model, jmap, djmap):
        """
        Plot the plain and differenced jmaps with the automatically tracked CME profiles overlaid.
        """
        fig, ax = self.plot_jmap(model, jmap, djmap)
        cme_profiles = self.track_cmes(model, djmap)
        print(cme_profiles)
        for cme in cme_profiles:
            print(cme)
            for key, val in cme.items():
                ax[1].plot(val['t'], val['e'], 'r.', label=key)

        return fig, ax

    def track_cmes(self, model, djmap):
        """
        Use image processing techniques to track the CME front in the differenced Jmap.
        """
        # Check that the model is compatible with this FoV.
        self._check_latitude_compatibility(model)

        # Check a ConeCME object exists.
        if not model.SURFlat[0].cmes:
            raise ValueError(
                "model.cmes is empty. Solve the model with at least one ConeCME before calling "
                "track_cmes.")
        if not all(isinstance(cme, s.ConeCME) for cme in model.SURFlat[0].cmes):
            raise TypeError(
                f"All entries in model.cmes must be instances of s.ConeCME. "
                f"Got types: {[type(cme).__name__ for cme in model.SURFlat[0].cmes]}."
            )

        times = model.SURFlat[0].time_out.to(u.day).value
        elons = self.e.to(u.deg).value

        # Clip and scale the jmap. Find ridges.
        djmap = np.nan_to_num(djmap, nan=0.0, posinf=0.0, neginf=0.0)
        vmin, vmax = np.nanpercentile(np.abs(djmap), [0, 100])
        djmap_clipped = np.clip(djmap, vmin, vmax)
        djmap_norm = (djmap_clipped - vmin) / (vmax - vmin)
        ridge = ski.filters.frangi(djmap_norm, sigmas=[1], black_ridges=False)
        ridge[ridge > np.percentile(ridge, 95)] = 1
        ridge[ridge < 1] = 0
        # Now do edge detection on the ridge.
        grad_t = ski.filters.sobel_v(djmap_norm)
        pos_grad = grad_t > 0

        # Build a mask to limit the search region for this CMEs t-e profile.
        # Base this on max/min elongation from propagating along the plane of the sky at v +/- dv
        cme_profiles = []
        for cme in model.SURFlat[0].cmes:
            # Only look for features that begin during CME injection
            t_launch = cme.t_launch.to(u.day).value
            r_min = cme.initial_height.to(u.m).value
            cme_mask = np.zeros(djmap.shape, dtype=bool)
            dt_pad = (2 * u.hour).to_value(u.s)
            for id_t, t in enumerate(times):
                travel_time = (t - t_launch) * 86400
                if travel_time < -dt_pad:
                    continue

                r_obs = self.position.r[id_t].to(u.m).value
                r_nose_fast = r_min + 1.25 * cme.v.to(u.m/u.s).value * (travel_time + dt_pad)
                r_nose_slow = r_min + 0.75 * cme.v.to(u.m / u.s).value * (travel_time - dt_pad)

                e_fast = np.rad2deg(np.arctan(r_nose_fast / r_obs))
                e_slow = np.rad2deg(np.arctan(r_nose_slow / r_obs))
                if e_fast > self.e_max.to(u.deg).value:
                    e_fast = self.e_max.to(u.deg).value

                if e_slow > self.e_max.to(u.deg).value:
                    e_slow = self.e_max.to(u.deg).value

                if e_slow == e_fast:
                    # Leave this loop, as CME has almost certainly left the field of view.
                    break

                id_good = (elons > e_slow) & (elons < e_fast)
                cme_mask[id_good, id_t] = True

            edges = ski.feature.canny(ridge, sigma=1, mask=cme_mask) & pos_grad
            label = ski.measure.label(edges)
            regions = ski.measure.regionprops(label)
            # Sort regions from largest to smallest (by number of coordinates)
            # This increases the chance the t-e profiles are in the correct order.
            regions = sorted(regions, key=lambda r: r.area, reverse=True)

            # For each region convert pixel coords to map coords and average multiple elons at fixed
            # times
            profiles = {}
            for id_r, region in enumerate(regions):

                if region.area < 5:
                    continue

                c = np.array(region.coords)
                # Scale pixel coords to map coords
                t_pix = times[0] + c[:, 1] * (times[-1] - times[0]) / times.size
                e_pix = elons[0] + c[:, 0] * (elons[-1] - elons[0]) / (elons.size)
                # Now average the e_pix values for each unique t_pix value.
                t_pix_unique = np.unique(t_pix)
                e_pix_mean = np.zeros(t_pix_unique.shape)
                for id_tu, tu in enumerate(t_pix_unique):
                    id_t = np.where(t_pix == tu)[0]
                    e_pix_mean[id_tu] = np.nanmean(e_pix[id_t])

                t_real = t_pix_unique + model.SURFlat[0].time_init.jd
                profiles[f"feature_{id_r:02d}"] = {'t': t_pix_unique, 't_real': t_real,
                                                   'e': e_pix_mean}

            cme_profiles.append(profiles)

        return cme_profiles


    def _density_interpolator(self, model, time_step):
        """
        Construct a 3D density interpolator from a :class:`SURF3d` model.

        The individual SURF runs in ``model.SURFlat`` are collated into a
        ``(radius, longitude, latitude)`` field.  Coordinates passed to the
        returned interpolator must have that same order and use SI/radian
        units (metres, radians, radians).
        """

        if not isinstance(model, s.SURF3d):
            raise TypeError("model must be an instance of SURF3d")
        if not model.SURFlat or any(not hasattr(run, "rho_grid") for run in model.SURFlat):
            raise ValueError("SURF3d must contain solved compressible SURF runs")

        reference = model.SURFlat[0]
        r = reference.r.to_value(u.m)
        lon = reference.lon.to_value(u.rad)
        lat = model.lat.to_value(u.rad)

        # Stack the latitude-plane fields. The resulting shape is
        # (latitude, radius, longitude).
        density = np.stack(
            [run.rho_grid[time_step] for run in model.SURFlat], axis=0
        )

        # Reorder the axes to (radius, longitude, latitude), as expected by
        # RegularGridInterpolator.
        density = density.transpose(1, 2, 0)

        # Convert proton mass density to electron number density, assuming
        # quasi-neutrality: n_e = n_p = rho / m_p.
        density = (density / const.m_p).to_value(u.m ** -3)

        # Pad longitude axis with wrap-around ghost cells to stop edge effects
        lon_pad = np.concatenate([[lon[-1] - 2.0 * np.pi], lon, [lon[0] + 2.0 * np.pi]])
        density_pad = np.concatenate([density[:, -1:, :], density, density[:, :1, :]], axis=1)

        interpolator = RegularGridInterpolator((r, lon_pad, lat),
                                               density_pad,
                                               method='linear',
                                               bounds_error=False,
                                               fill_value=np.nan)

        return interpolator

    def _fov_mask(self, model):
        """
        Return a mask for the imager field of view, with points outside the model domain set True
        """
        fov_mask = np.full_like(self.e_grid.value, False, dtype=bool)
        r = model.SURFlat[0].r.to(u.solRad).value.copy()
        lon = model.SURFlat[0].lon.to(u.rad).value.copy()
        lat = model.lat.to(u.rad).value.copy()

        # Find bad radii
        r_img = self.r_grid.to(u.solRad).value.copy()
        id_bad_r = (r_img < r.min()) | (r_img > r.max())

        # A limited SURF longitude grid may straddle zero.  Infer the solved
        # arc from the largest circular gap between its grid points, rather
        # than moving the longitude discontinuity to an imager-dependent
        # location.
        lon = np.sort(np.mod(lon, 2.0 * np.pi))
        lon_imgr = self.lon_grid.to_value(u.rad)
        if lon.size < 2:
            # A one-longitude model has support only at its sampled longitude.
            id_bad_lon = ~np.isclose(
                np.mod(lon_imgr - lon[0], 2.0 * np.pi), 0.0,
            )
        else:
            gaps = np.diff(np.concatenate((lon, [lon[0] + 2.0 * np.pi])))
            if np.allclose(gaps, gaps[0]):
                # Uniform gaps mean that the grid spans the full circle.
                id_bad_lon = np.zeros(self.lon_grid.shape, dtype=bool)
            else:
                gap_index = np.argmax(gaps)
                lon_start = lon[(gap_index + 1) % lon.size]
                lon_stop = lon[gap_index]
                span = np.mod(lon_stop - lon_start, 2.0 * np.pi)
                offset = np.mod(lon_imgr - lon_start, 2.0 * np.pi)
                id_bad_lon = offset > span

        # Find bad latitudes.
        lat_img = self.lat_grid.to(u.rad).value.copy()
        id_bad_lat = (lat_img < lat.min()) | (lat_img > lat.max())

        # Join the bad values.
        fov_mask[id_bad_r | id_bad_lon | id_bad_lat] = True

        return fov_mask

    def _check_latitude_compatibility(self, model):
        """Ensure the 3D model latitude range covers the imager FOV."""
        if not isinstance(model, s.SURF3d):
            raise TypeError("model must be an instance of SURF3d")
        self._compute_fov_geometry(0)
        if (self.lat_grid.min() < model.lat.min() or
                self.lat_grid.max() > model.lat.max()):
            raise ValueError("SURF3d latitude range does not cover the imager FOV")



    def plot_los_3d(self, time_step, ert):
        """Plot a line of sight in heliocentric spherical-polar coordinates.

        The LOS is calculated in Helioprojective Radial coordinates and transformed
        to Heliographic Stonyhurst by :meth:`_compute_fov_geometry`.  The returned
        Matplotlib axes use heliocentric Cartesian coordinates in solar radii; the
        spherical-polar coordinates are ``(r, longitude, latitude)``.

        Parameters
        ----------
        time_step : int
            Observer ephemeris index.

        Returns
        -------
        tuple
            ``(fig, ax)`` containing the plot.
        """
        self._compute_fov_geometry(time_step)

        # Observer Cartesian coords
        x_o, y_o, z_o = spherical_to_cartesian(self.position.r[time_step],
                                               self.position.lat[time_step],
                                               self.position.lon[time_step])

        x_o = x_o.to(u.solRad)
        y_o = y_o.to(u.solRad)
        z_o = z_o.to(u.solRad)

        # Observer Cartesian coords
        x_e, y_e, z_e = spherical_to_cartesian(ert.r[time_step],
                                               ert.lat[time_step],
                                               ert.lon[time_step])

        x_e = x_e.to(u.solRad)
        y_e = y_e.to(u.solRad)
        z_e = z_e.to(u.solRad)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter([0], [0], [0], color="gold", edgecolor="black", s=80, label="Sun")
        ax.scatter(x_o, y_o, z_o, color="tab:red", s=50, label="Observer")
        ax.scatter(x_e, y_e, z_e, color="tab:blue", s=50, label="Earth")

        # LOS Cartesian coords
        x, y, z = spherical_to_cartesian(self.r_grid[::20, ::10],
                                         self.lat_grid[::20, ::10],
                                         self.lon_grid[::20, ::10])

        x = x.to(u.solRad)
        y = y.to(u.solRad)
        z = z.to(u.solRad)

        ax.plot(x, y, z, "k.", markersize=0.3)

        ax.set_xlabel("x [R$_\\odot$]")
        ax.set_ylabel("y [R$_\\odot$]")
        ax.set_zlabel("z [R$_\\odot$]")
        ax.set_xlim(-260, 260)
        ax.set_ylim(-260, 260)
        ax.set_zlim(-30, 30)
        ax.legend()
        ax.view_init(azim=0, elev=90, roll=0)
        return (fig, ax)


def compute_target_hpr_coords(observer, target):
    """
    Compute the position angle of a target relative to the observer. Both the observer and target
    must be instances of the Observer class that span the same time range.

    Args:
        observer: An instance of the Observer class
        target: An instance of the Observer class

    Returns:
        psi: The position angle of the target relative to the observer.
    """

    # Get skycoord obj for the observer.
    psi = np.zeros(observer.time.size)
    elon = np.zeros(observer.time.size)

    for i in range(observer.time.size):

        observer_heeq = SkyCoord(
            lon=observer.lon[i],
            lat=observer.lat[i],
            radius=observer.r[i],
            obstime=observer.time[i],
            frame=frames.HeliographicStonyhurst
        )

        target_heeq = SkyCoord(
            lon=target.lon[i],
            lat=target.lat[i],
            radius=target.r[i],
            obstime=target.time[i],
            frame=frames.HeliographicStonyhurst
        )

        # Convert the HEEQ CME coords to HPR.
        target_hpr = target_heeq.transform_to(
            frames.HelioprojectiveRadial(
                observer=observer_heeq,
                obstime=target_heeq.obstime
            )
        )
        
        psi[i] = target_hpr.psi.to(u.deg).value
        elon[i] = target_hpr.theta.to(u.deg).value

    psi = psi * u.deg
    elon = elon * u.deg
    return psi, elon


if __name__ == "__main__":

    ################################################################################################
    # Need to know how many timesteps and ephemeris of the observer. So make a dummy model.
    t_start = datetime(2026,1,1)
    cr, cr_lon_init = sIN.datetime2surfinputs(t_start)
    print(cr, cr_lon_init)

    demo_dir = s._setup_dirs_()['example_inputs']
    wsafilepath = os.path.join(demo_dir, '2022-02-24T22Z.wsa.gong.fits')
    vr_map, vr_longs, vr_lats, _, _, _, _ = sIN.get_WSA_maps(wsafilepath)

    # Setup SURF
    v_in = np.ones(128) * 400 * u.km / u.s
    model = s.SURF(v_boundary=v_in,
                   cr_num=cr, cr_lon_init=cr_lon_init,
                   r_min=21.5 * u.solRad, r_max=240 * u.solRad,
                   lon_start=270 * u.deg, lon_stop=90 * u.deg,
                   simtime=3.0 * u.day, dt_scale=4.0,
                   solver='hydro', track_cmes=False)

    sta = model.get_observer('STA')
    ert = model.get_observer('EARTH')
    ################################################################################################
    # Create SyntheticImager instance along Earth's PA from STA.

    # Compute the position angle of Earth from STA.
    psi_ert, elon_ert = compute_target_hpr_coords(sta, ert)
    psi_ert_avg = np.mean(psi_ert.to(u.deg).value)
    print(f"Earth position angle: {psi_ert_avg}")

    # Set up the synthetic imager for STA and along Earth's PA.
    imgr = SyntheticImager(sta, pa=psi_ert_avg, elon_min=5.0, elon_max=30.0)

    ################################################################################################
    # Now we can initiliase SURF3d

    # Get the latitude range spanned by the SyntheticImager's field of view.
    # This is to help set up the SURF3d latitudes.
    imgr._compute_fov_geometry(0)
    lat_min = imgr.lat_grid.min().to(u.deg)
    lat_max = imgr.lat_grid.max().to(u.deg)
    print(f"Lat lims:{lat_min} {lat_max}")

    # Set up SURF3d using this latitude range
    dl = 2 * u.deg
    model3d = s.SURF3d(v_map=vr_map,
                       v_map_lat=vr_lats, v_map_long=vr_longs,
                       cr_num=cr, cr_lon_init=cr_lon_init,
                       latitude_max=lat_max + dl, latitude_min=lat_min - dl,
                       r_min=21.5 * u.solRad, r_max=240 * u.solRad,
                       lon_start=270 * u.deg, lon_stop=90 * u.deg,
                       simtime=3.0 * u.day, dt_scale=4.0,
                       solver='hydro', track_cmes=False)

    # Run the model with a CME
    cme = s.ConeCME(t_launch=0.25 * u.day, longitude=0.0 * u.deg, latitude=0 * u.deg,
                    width=50 * u.deg, v=800 * (u.km / u.s), thickness=0 * u.solRad,
                    initial_height=21.5 * u.solRad)

    cme2 = s.ConeCME(t_launch=1 * u.day, longitude=-30.0 * u.deg, latitude=10 * u.deg,
                    width=50 * u.deg, v=600 * (u.km / u.s), thickness=0 * u.solRad,
                    initial_height=21.5 * u.solRad)

    model3d.solve([cme, cme2])

    # Use the SURF3d solution to make a Jmap.
    jmap, djmap = imgr.compute_jmap(model3d)
    fig, ax = imgr.plot_jmap(model3d, jmap, djmap)
    plt.show()

    fig, ax = imgr.plot_jmap_with_cme_profiles(model3d, jmap, djmap)
    plt.show()
