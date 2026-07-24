"""
This module contains the SyntheticImager class, which generates synthetic heliospheric images from
SURF model output.
"""

import astropy.constants as const
import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.integrate import trapezoid
import skimage as ski

from surf import surf as s

class SyntheticImager:
    """
    Class to generate synthetic heliospheric images from SURF model output.
    """

    def __init__(self, observer, elon_min=5.0, elon_max=30.0, elon_sign=1):
        """
        Parameters
        ----------
        observer : Observer
            The observer ephemeris object.
        elon_min : float, optional
            Minimum elongation of the field of view, in degrees. Default is 5.0.
            May be negative to indicate a field of view on the opposite side of the
            observer (i.e. looking in the retrograde direction).
        elon_max : float, optional
            Maximum elongation of the field of view, in degrees. Default is 30.0.
            Must be greater than elon_min, and both values must have the same sign.
        elon_sign : int, optional
            Sign of the elongation, controlling which side of the observer the FOV
            is on. Use +1 (default) for the prograde (forward-looking) FOV, or -1
            for the retrograde (backward-looking) FOV.
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
        if elon_sign not in (1, -1):
            raise ValueError(
                f"elon_sign ({elon_sign}) must be either +1 (prograde) or -1 (retrograde)."
            )

        self.sigma_e = 7.95 * 10 ** (-30) * u.m ** 2 / u.steradian
        self.u_ld = 0.63  # Limb darkening coefficient. For 5500 angstroms.

        # Store the sign so geometry routines know which side to look
        self._elon_sign = elon_sign

        # Store full observer ephemeris
        self.observer_times = observer.time
        self.observer_rs = observer.r.to(u.m)
        self.observer_lons = observer.lon.to(u.rad)
        self.observer_lats = observer.lat.to(u.rad)

        # Set up elongation arrays (time-independent)
        self.e_min, self.e_max, self.e, self.de = self.elon_grid(elon_min, elon_max)

        # Set up LOS distance arrays (time-independent)
        self.z_min, self.z_max, self.z, self.dz = self.los_distance_grid()

        # Create mesh grid as these don't change with time
        self.e_grid, self.z_grid = np.meshgrid(self.e, self.z)

        # Initialise FOV geometry using the first time step
        self._compute_fov_geometry(0)

    def _check_latitude_compatibility(self, model, lat_tolerance_deg=3.0):
        """
        Raise a ValueError if the observer latitude differs from the model latitude
        by more than lat_tolerance_deg degrees.

        Parameters
        ----------
        model : SURF model instance
            The model whose latitude will be compared against the observer latitude.
        lat_tolerance_deg : float
            Maximum permitted absolute difference in latitude, in degrees.
        """

        lat_diffs = (self.observer_lats - model.latitude).to(u.deg).value
        max_lat_diff = np.max(np.abs(lat_diffs))
        if max_lat_diff > lat_tolerance_deg:
            raise ValueError(
                f"Observer latitude differs from model latitude by more than "
                f"{lat_tolerance_deg:.2f} deg. The SyntheticImager assumes the observer and "
                f"model share the same latitudinal plane."
            )

    def _compute_fov_geometry(self, time_step):
        """
        Compute all field-of-view geometry quantities for the observer position
        at the given time step index.

        Updates instance attributes: observer_time, observer_lon, observer_r,
        observer_x, observer_y, omega, e_grid, z_grid, r_grid, chi_grid,
        lon_grid, x_grid, y_grid, g, gr, gt, gp.
        """

        # Observer position at this time step
        self.observer_time = self.observer_times[time_step]
        self.observer_lon = self.observer_lons[time_step]
        self.observer_lat = self.observer_lats[time_step]
        self.observer_r = self.observer_rs[time_step]
        self.observer_x = self.observer_r * np.cos(self.observer_lon)
        self.observer_y = self.observer_r * np.sin(self.observer_lon)

        # Compute the angular halfwidth of the Sun from the observer's position
        self.omega = self.compute_omega()

        # Heliocentric radius of each LOS element - using cosine rule
        B = self.observer_r.to(u.m)
        C = self.z_grid.to(u.m)
        self.r_grid = np.sqrt(B ** 2 + C ** 2 - (2.0 * B * C * np.cos(self.e_grid)))

        # Scattering angle of radial illumination at each LOS - using cosine rule
        self.chi_grid = np.arccos((C ** 2 + self.r_grid ** 2 - B ** 2) / (2.0 * C * self.r_grid))

        # Find the angle that completes the observer-scattering site-sun triangle
        theta_ls = np.pi * u.rad - self.e_grid - self.chi_grid

        # Compute heliolongitude of each LOS element.
        # For a prograde FOV (elon_sign = +1) the LOS fans out ahead of the observer,
        # so longitude decreases (for observer_lon < pi) or increases (for observer_lon > pi).
        # For a retrograde FOV (elon_sign = -1) the LOS fans out behind the observer,
        # so the longitude offset is flipped.
        if self._elon_sign > 0:
            # Prograde: forward-looking FOV (original behaviour)
            if self.observer_lon < np.pi * u.rad:
                self.lon_grid = self.observer_lon - theta_ls
            else:
                self.lon_grid = theta_ls - (2 * np.pi * u.rad - self.observer_lon)
        else:
            # Retrograde: backward-looking FOV — flip the longitude offset
            if self.observer_lon < np.pi * u.rad:
                self.lon_grid = self.observer_lon + theta_ls
            else:
                self.lon_grid = (2 * np.pi * u.rad - self.observer_lon) - theta_ls

        # Make sure longitudes are between 0 and 2pi
        self.lon_grid = np.where(self.lon_grid.value < 0, self.lon_grid.value + 2 * np.pi,
                                 self.lon_grid.value)
        self.lon_grid = self.lon_grid * u.rad

        # Get the Heliocentric Cartesian coordinates of each LOS element
        self.x_grid = self.r_grid * np.cos(self.lon_grid)
        self.y_grid = self.r_grid * np.sin(self.lon_grid)

        # Compute the geometric factor for each LOS element
        self.g, self.gr, self.gt, self.gp = self.compute_ts_intensity_factors()

    def compute_omega(self):
        """
        Compute the angular halfwidth of the Sun from the observer's position.
        Returns:
            omega: astropy.units.Quantity with units of radians, giving the angular halfwidth of
                   the Sun from the observer's position.
        """

        # Sun center to observer vector:
        omega = np.arcsin(const.R_sun / self.observer_r)
        return omega

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
        dz = 0.1 * u.solRad.to(u.m) * u.m
        z_min = dz.copy()
        z_max = 2.0 * u.AU.to(u.m) * u.m
        dz = 0.35 * u.solRad.to(u.m) * u.m
        z = np.arange(z_min.value, z_max.value + dz.value, dz.value) * dz.unit

        return z_min, z_max, z, dz

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

        # Minimum elongation permitted by the model inner boundary
        elon_min_model = np.arcsin(
            model.r[0].to(u.solRad).value / self.observer_r.to(u.solRad).value)

        for i in range(I_fov.shape[1]):

            elon = self.e[i]

            # Skip LoS that cross the model inner boundary — leave columns as NaN
            if elon.to(u.rad).value < elon_min_model:
                continue

            n_e_los = self.get_los_density(interpolator, elon, replace_nans=True)

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
        coords = [self.r_grid[:, id_elon].value.ravel(), self.lon_grid[:, id_elon].value.ravel()]
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

        jmap = np.full((self.e.size, model.time_out.size), np.nan)

        for time_step in range(model.time_out.size):
            # compute_total_intensity -> compute_all_los_intensity_profile ->
            # _compute_fov_geometry updates geometry for this time step
            I = self.compute_total_intensity(model, time_step)
            jmap[:, time_step] = I

            # Mask elongations that cross the model inner boundary
            elon_min_model = np.arcsin(
                model.r[0].to(u.solRad).value / self.observer_r.to(u.solRad).value)
            invalid = self.e.to(u.rad).value < elon_min_model
            jmap[invalid, time_step] = np.nan

        djmap = np.diff(jmap, axis=1, prepend=jmap[:, :1])

        return jmap, djmap

    def plot_jmap(self, model, jmap, djmap):
        """
        Make a 2-panel plot of the normal and difference image jmaps
        """

        times = model.time_out.to(u.day).value
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
        for cme in cme_profiles:
            for key, val in cme.items():
                if key == 'feature_00':
                    ax[1].plot(val['t'], val['e'], 'r.', label=key)

        return fig, ax

    def track_cmes(self, model, djmap):
        """
        Use image processing techniques to track the CME front in the differenced Jmap.
        """
        # Check that the model is compatible with this FoV.
        self._check_latitude_compatibility(model)

        # Check a ConeCME object exists.
        if not model.cmes:
            raise ValueError(
                "model.cmes is empty. Solve the model with at least one ConeCME before calling "
                "track_cmes.")
        if not all(isinstance(cme, s.ConeCME) for cme in model.cmes):
            raise TypeError(
                f"All entries in model.cmes must be instances of s.ConeCME. "
                f"Got types: {[type(cme).__name__ for cme in model.cmes]}."
            )

        times = model.time_out.to(u.day).value
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

        cme_profiles = []
        for cme in model.cmes:

            # USE TRACER PARTICLES TO ISOLATE THE CME IN TIME - ELONGATION SPACE.
            flank, extent = self.compute_flank_profile(cme)

            cme_mask = np.zeros(djmap.shape, dtype=bool)
            for id_t in extent.index:
                e_min = extent.loc[id_t, 'e_min']
                e_max = extent.loc[id_t, 'e_max']
                if np.isnan(e_min):
                    e_min = 0.0
                if np.isnan(e_max):
                    continue
                else:
                    id_e = (elons >= e_min) & (elons <= e_max + 5)
                    cme_mask[id_e, id_t] = True

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

                t_real = t_pix_unique + model.time_init.jd
                profiles[f"feature_{id_r:02d}"] = {'t': t_pix_unique, 't_real': t_real,
                                                   'e': e_pix_mean}

            cme_profiles.append(profiles)

        return cme_profiles

    def compute_flank_profile(self, cme):
        """
        Compute the time elongation profile of the flank of a ConeCME in HUXt. The observer
        longtidue is specified relative to Earth but otherwise matches Earth's coords.

        Parameters
        ----------
        cme: A ConeCME object from a completed HUXt run (i.e the ConeCME.coords dictionary has
        been populated).
        Returns
        -------
        obs_profile: Pandas dataframe giving the coordinates of the ConeCME flank from
                     the observers perspective, including the time, elongation, position angle,
                     and HEEQ radius and longitude.
        """
        times = np.array(
            [coord['model_time'].to(u.day).value for i, coord in cme.coords.items()]) * u.day

        # Compute observers location using earth ephem, adding on observers longitude offset from Earth
        # and correct for runover 2*pi
        flank = pd.DataFrame(index=np.arange(times.size), columns=['time', 'el', 'r', 'lon'])
        flank['time'] = times.to(u.day).value

        extent = pd.DataFrame(index=np.arange(times.size), columns=['time', 'e_min', 'e_max'])
        extent['time'] = times.to(u.day).value

        # Check that the time arrays match in length
        if self.observer_times.size != len(times):
            raise ValueError(f"The number of times in the ConeCME ({len(times)}) does not match "
                             f"the  number of times in the observer ({self.observer_times.size}).")

        for i, coord in cme.coords.items():

            if len(coord['r']) == 0:
                flank.loc[i, ['lon', 'r', 'el']] = np.nan
                continue

            r_obs = self.observer_rs[i].to(u.m)
            lon_obs = self.observer_lons[i].to(u.rad)
            lat_obs = self.observer_lats[i].to(u.rad)
            x_obs = r_obs * np.cos(lat_obs) * np.cos(lon_obs)
            y_obs = r_obs * np.cos(lat_obs) * np.sin(lon_obs)
            z_obs = r_obs * np.sin(lat_obs)

            lon_cme = coord['lon'].to(u.rad)
            lat_cme = coord['lat'].to(u.rad)
            r_cme = coord['r'].to(u.m)

            x_cme = r_cme * np.cos(lat_cme) * np.cos(lon_cme)
            y_cme = r_cme * np.cos(lat_cme) * np.sin(lon_cme)
            z_cme = r_cme * np.sin(lat_cme)
            #############
            # Compute the observer CME distance, S, and elongation

            x_cme_s = x_cme - x_obs
            y_cme_s = y_cme - y_obs
            z_cme_s = z_cme - z_obs
            s = np.sqrt(x_cme_s ** 2 + y_cme_s ** 2 + z_cme_s ** 2)

            numer = (r_obs ** 2 + s ** 2 - r_cme ** 2).value
            denom = (2.0 * r_obs * s).value
            e_obs = np.arccos(numer / denom)

            # Restrict those CME points to those in FOV
            # For those ahead of Earth, this is negative y_cme_s
            # For those behind Earth, this is positive y_cme_s
            if self.observer_lons[i] < np.pi * u.rad:
                id_sub = y_cme_s.value < 0
                e_obs = e_obs[id_sub]
                lon_cme = lon_cme[id_sub]
                r_cme = r_cme[id_sub]
            elif self.observer_lons[i] > np.pi * u.rad:
                id_sub = y_cme_s.value > 0
                e_obs = e_obs[id_sub]
                lon_cme = lon_cme[id_sub]
                r_cme = r_cme[id_sub]

            # Find the flank coordinate and update output
            id_obs_flank = np.argmax(e_obs)
            flank.loc[i, 'lon'] = lon_cme[id_obs_flank].value
            flank.loc[i, 'r'] = r_cme[id_obs_flank].value
            flank.loc[i, 'el'] = np.rad2deg(e_obs[id_obs_flank])

            # Find the flank coordinate and update output
            extent.loc[i, 'e_min'] = np.rad2deg(np.nanmin(e_obs))
            extent.loc[i, 'e_max'] = np.rad2deg(np.nanmax(e_obs))

        # Force values to be floats.
        keys = ['time', 'lon', 'r', 'el']
        flank[keys] = flank[keys].astype(np.float64)

        keys = ['time', 'e_min', 'e_max']
        extent[keys] = extent[keys].astype(np.float64)
        return flank, extent

    def _density_interpolator(self, model, time_step):
        """
        Construct and return a RegularGridInterpolator object for the density field.
        """

        # Interpolate the model density field into the FOV coordinates.
        # Extract electron density from SURF model at a specific time index

        r = model.r.to(u.m).value
        lon = model.lon.to(u.rad).value

        # Convert proton mass density to electron number density
        # Assuming quasi-neutrality: n_e = n_p = rho / m_p
        density = model.rho_grid[time_step, :, :]
        density = (density / const.m_p).value  # electrons per m^3

        # Pad longitude axis with wrap-around ghost cells to stop edge effects
        lon_pad = np.concatenate([[lon[-1] - 2.0 * np.pi], lon, [lon[0] + 2.0 * np.pi]])
        density_pad = np.concatenate([density[:, -1:], density, density[:, :1]], axis=1)

        interpolator = RegularGridInterpolator((r, lon_pad),
                                               density_pad,
                                               method='linear',
                                               bounds_error=False,
                                               fill_value=np.nan)

        return interpolator

    def compute_fov_patch(self, time_step):
        """Compute a patch showing the synthetic imager field of view for overlaying on plots"""

        self._compute_fov_geometry(time_step)

        # Build the patch boundary by sampling along all four edges of the FOV grid.
        # Edge indices: inner arc (col 0), outer arc (col -1), and the two sides (rows 0 and -1).
        # All values are converted to solar radii for plotting.

        # Inner arc: first elongation column, all z rows (near boundary)
        r_inner_arc = self.r_grid[:, 0].to(u.solRad).value
        l_inner_arc = self.lon_grid[:, 0].to(u.rad).value

        # Outer arc: last elongation column, all z rows (far boundary), reversed
        r_outer_arc = self.r_grid[:, -1].to(u.solRad).value[::-1]
        l_outer_arc = self.lon_grid[:, -1].to(u.rad).value[::-1]

        # Side 1: first z row (smallest elongation side), inner to outer
        r_side1 = self.r_grid[0, :].to(u.solRad).value
        l_side1 = self.lon_grid[0, :].to(u.rad).value

        # Side 2: last z row (largest elongation side), outer to inner, reversed
        r_side2 = self.r_grid[-1, :].to(u.solRad).value[::-1]
        l_side2 = self.lon_grid[-1, :].to(u.rad).value[::-1]

        # Concatenate all edges into a closed polygon boundary
        r_patch = np.concatenate([r_inner_arc, r_side1, r_outer_arc, r_side2])
        l_patch = np.concatenate([l_inner_arc, l_side1, l_outer_arc, l_side2])

        # For a polar plot, vertices are (theta, r) = (longitude, radius)
        vertices = np.column_stack([l_patch, r_patch])
        patch = plt.Polygon(vertices, closed=True, fill=True,
                            edgecolor='white', linewidth=1.5, linestyle='-', alpha=0.25)
        return patch

    def compute_los_coords(self, time_step, elon):
        """Compute the longitude and radius coords of the line of sight at a given elongation."""

        self._compute_fov_geometry(time_step)

        id_elon = np.argmin(np.abs(self.e - elon))

        r_inner = self.r_grid[0, id_elon].to(u.solRad).value
        l_inner = self.lon_grid[0, id_elon].to(u.rad).value

        r_outer = self.r_grid[-1, id_elon].to(u.solRad).value
        l_outer = self.lon_grid[-1, id_elon].to(u.rad).value

        return np.array([r_inner, r_outer]) * u.solRad, np.array([l_inner, l_outer]) * u.rad
