from astropy.stats import biweight_location
from functools import partial
import george
from george import kernels
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize
# from astropy.stats import sigma_clip
# from scipy.interpolate import UnivariateSpline


'''
2D GP fitting: Cited from Avocado: (Boone et al. 2021)

'''

# Central wavelengths for each band.
band_central_wavelengths = {
    "lsstu": 3671.0,
    "lsstg": 4827.0,
    "lsstr": 6223.0,
    "lssti": 7546.0,
    "lsstz": 8691.0,
    "lssty": 9710.0,
    "ztfg": 4813.9,
    "ztfr": 6421.8,
    "ztfi": 7883.0,
}

# Colors for plotting
band_plot_colors = {
    "lsstu": "C6",
    "lsstg": "C4",
    "lsstr": "C0",
    "lssti": "C2",
    "lsstz": "C3",
    "lssty": "goldenrod",


    "ztfg": "green",
    "ztfr": "red",
    "ztfi": "C1",
}

# Markers for plotting
band_plot_markers = {
    "lsstu": "o",
    "lsstg": "v",
    "lsstr": "^",
    "lssti": "<",
    "lsstz": ">",
    "lssty": "s",

    "ztfg": "o",
    "ztfr": "^",
    "ztfi": "v",
}


def get_band_central_wavelength(band):
    """Return the central wavelength for a given band.

    If the band does not yet have a color assigned to it, an AvocadoException
    is raised.

    Parameters
    ----------
    band : str
        The name of the band to use.
    """
    if band in band_central_wavelengths:
        return band_central_wavelengths[band]
    else:
        return None


def get_band_plot_color(band):
    """Return the plot color for a given band.

    If the band does not yet have a color assigned to it, then a random color
    will be assigned (in a systematic way).

    Parameters
    ----------
    band : str
        The name of the band to use.
    """
    if band in band_plot_colors:
        return band_plot_colors[band]

    print("No plot color assigned for band %s, assigning a random one." % band)

    # Systematic random colors. We use the hash of the band name.
    # Note: hash() uses a random offset in python 3 so it isn't consistent
    # between runs!
    import hashlib

    hasher = hashlib.md5()
    hasher.update(band.encode("utf8"))
    hex_color = "#%s" % hasher.hexdigest()[-6:]

    band_plot_colors[band] = hex_color

    return hex_color


def get_band_plot_marker(band):
    """Return the plot marker for a given band.

    If the band does not yet have a marker assigned to it, then we use the
    default circle.

    Parameters
    ----------
    band : str
        The name of the band to use.
    """
    if band in band_plot_markers:
        return band_plot_markers[band]
    else:
        return "o"


class AstronomicalObject:
    """An astronomical object, with metadata and a lightcurve.

    An astronomical object has both metadata describing its global properties,
    and observations of its light curve.

    Parameters
    ----------
    metadata : dict-like
        Metadata for this object. This is represented using a dict
        internally, and must be able to be cast to a dict. Any keys and
        information are allowed. Various functions assume that the
        following keys exist in the metadata:

        - object_id: A unique ID for the object. This will be stored as a
          string internally.
        - galactic: Whether or not the object is in the Milky Way galaxy or
          not.
        - host_photoz: The photometric redshift of the object's host galaxy.
        - host_photoz_error: The error on the photometric redshift of the
          object's host galaxy.
        - host_specz: The spectroscopic redshift of the object's host galaxy.

        For training data objects, the following keys are assumed to exist in
        the metadata:
        - redshift: The true redshift of the object.
        - class: The true class label of the object.

    observations : pandas.DataFrame
        Observations of the object's light curve. This should be a pandas
        DataFrame with at least the following columns:

        - time: The time of each observation.
        - band: The band used for the observation.
        - flux: The measured flux value of the observation.
        - flux_error: The flux measurement uncertainty of the observation.
    """

    def __init__(self, observations, object_id):
        """Create a new AstronomicalObject"""
        self.observations = observations
        self.object_id = object_id
        self.executed = True
        # self.detrend_and_clip()
        self.mean_g, self.mean_r, self.std_g, self.std_r = self.normalize_param
        if self.observations.empty:
            print('No valid observations after removing outliers.')
            self.executed = False
            return None
        
        if self.executed:
            self.convert_mag_to_flux()
        

        self._default_gaussian_process = None


    @property
    def bands(self):
        """Return a list of bands that this object has observations in

        Returns
        -------
        bands : numpy.array
            A list of bands, ordered by their central wavelength.
        """
        unsorted_bands = np.unique(self.observations["band"])
        sorted_bands = np.array(sorted(unsorted_bands, key=get_band_central_wavelength))
        return sorted_bands

    def convert_pred_to_mag(self, predictions):
        """Convert the flux to magnitude.
        """
        return -2.5 * np.log10(predictions) + 23.9
    
    def convert_pred_err_to_mag_err(self, predictions, pred_err):
        """Convert the flux error to magnitude error.
        """
        return 2.5  * pred_err / (predictions * np.log(10)) 

    def convert_mag_to_flux(self):
        """Convert the magnitude to flux.
        """
        self.observations["flux"] = 10**((self.observations["mag"] - 23.9)/(-2.5))

        self.observations["flux_error"] = np.abs(self.observations["mag_err"] * self.observations["flux"] * np.log(10) / 2.5)
        return self.observations


    @property
    def normalize_param(self):
        mean_g = np.mean(self.observations[self.observations["band"] == "ztfg"]["mag"])
        mean_r = np.mean(self.observations[self.observations["band"] == "ztfr"]["mag"])
        std_g = np.std(self.observations[self.observations["band"] == "ztfg"]["mag"])
        std_r = np.std(self.observations[self.observations["band"] == "ztfr"]["mag"])
     
        return mean_g, mean_r, std_g, std_r
    
    # def detrend_and_clip(self, spline_smooth=1e-2, sigma=3):

    #     self.observations = self.observations.dropna()
    
    #     for band in ['ztfg', 'ztfr']:
        
    #         band_data = self.observations.loc[self.observations["band"] == band, "mag"]
    #         band_mjd = self.observations.loc[self.observations["band"] == band, "time"]
    #         if len(band_mjd) > 0:
    #             time_diff = np.diff(band_mjd)
    #             time_diff = np.append(time_diff, 0)  # Aligns length with band_mjd
    #             mask = time_diff <= 80
    #             band_data = band_data.iloc[mask]
    #             band_mjd = band_mjd.iloc[mask]  # Keep time and mag in sync
               
    #         # Check if we have enough data points for spline fitting (minimum 4 for cubic spline)
    #         if len(band_data) < 4:
    #             print(f'Warning: Not enough data points ({len(band_data)}) for spline fitting in band {band}. Skipping detrending.')
    #             if len(band_data) <= 1:
    #                 print('remove this band data: ', band)
    #                 self.observations = self.observations[self.observations["band"] != band]
    #             continue
            
    #         # Sort data by time to ensure x is increasing
    #         sort_idx = np.argsort(band_mjd)
    #         band_mjd = band_mjd.iloc[sort_idx]
    #         band_data = band_data.iloc[sort_idx]
            
    #         spline = UnivariateSpline(band_mjd, band_data, s=spline_smooth * len(band_mjd))
            
    #         trend = spline(band_mjd)
    #         residuals = band_data - trend
    #         clipped = sigma_clip(residuals, sigma=sigma)
    #         mask = ~clipped.mask
        
    #         masked_band_data = band_data[mask]
    #         masked_band_mjd = band_mjd[mask]
         
    #         self.observations.loc[self.observations["band"] == band, "mag"] = masked_band_data
    #         self.observations.loc[self.observations["band"] == band, "time"] = masked_band_mjd


    #     # Drop any rows that became NaN after detrending/clipping
    #     self.observations = self.observations.dropna()

    #     self.observations = self.observations.reset_index(drop=True)


    #     self.mean_g, self.mean_r, self.std_g, self.std_r = self.normalize_param
  
    #     print('Outliers removed by detrending and clipping.')



    def subtract_background(self):
        """Subtract the background levels from each band.

        The background levels are estimated using a biweight location
        estimator. This estimator will calculate a robust estimate of the
        background level for objects that have short-lived light curves, and it
        will return something like the median flux level for periodic or
        continuous light curves.

        Returns
        -------
        subtracted_observations : pandas.DataFrame
            A modified version of the observations DataFrame with the
            background level removed.
        """
        subtracted_observations = self.observations.copy()

        self.ref_flux = {}
        for band in self.bands:
            mask = self.observations["band"] == band
            band_data = self.observations[mask]

            # Use a biweight location to estimate the background
            ref_flux = biweight_location(band_data["flux"])
            self.ref_flux[band] = ref_flux

            subtracted_observations.loc[mask, "flux"] -= ref_flux

        return subtracted_observations


    def preprocess_observations(self, subtract_background=True, **kwargs):
        """Apply preprocessing to the observations.

        This function is intended to be used to transform the raw observations
        table into one that can actually be used for classification. For now,
        all that this step does is apply background subtraction.

        Parameters
        ----------
        subtract_background : bool (optional)
            If True (the default), a background subtraction routine is applied
            to the lightcurve before fitting the GP. Otherwise, the flux values
            are used as-is.
        kwargs : dict
            Additional keyword arguments. These are ignored. We allow
            additional keyword arguments so that the various functions that
            call this one can be called with the same arguments, even if they
            don't actually use them.

        Returns
        -------
        preprocessed_observations : pandas.DataFrame
            The preprocessed observations that can be used for further
            analyses.
        """

        if not self.executed:
            return None
        
        if subtract_background:
            preprocessed_observations = self.subtract_background()
        else:
            preprocessed_observations = self.observations

            
        return preprocessed_observations

    def fit_gaussian_process(
        self,
        fix_scale=False,
        verbose=False,
        guess_length_scale=20.0,
        **preprocessing_kwargs
    ):
        """Fit a Gaussian Process model to the light curve.

        We use a 2-dimensional Matern kernel to model the transient. The kernel
        width in the wavelength direction is fixed. We fit for the kernel width
        in the time direction as different transients evolve on very different
        time scales.

        Parameters
        ----------
        fix_scale : bool (optional)
            If True, the scale is fixed to an initial estimate. If False
            (default), the scale is a free fit parameter.
        verbose : bool (optional)
            If True, output additional debugging information.
        guess_length_scale : float (optional)
            The initial length scale to use for the fit. The default is 20
            days.
        preprocessing_kwargs : kwargs (optional)
            Additional preprocessing arguments that are passed to
            `preprocess_observations`.

        Returns
        -------
        gaussian_process : function
            A Gaussian process conditioned on the object's lightcurve. This is
            a wrapper around the george `predict` method with the object flux
            fixed.
        gp_observations : pandas.DataFrame
            The processed observations that the GP was fit to. This could have
            effects such as background subtraction applied to it.
        gp_fit_parameters : list
            A list of the resulting GP fit parameters.
        """

        if not self.executed:
            return None, None, None
        
        gp_observations = self.preprocess_observations(**preprocessing_kwargs)
        if gp_observations is None:
            return None, None, None

        fluxes = gp_observations["flux"]
        flux_errors = gp_observations["flux_error"]

        wavelengths = gp_observations["band"].map(get_band_central_wavelength)
        times = gp_observations["time"]

        # Use the highest signal-to-noise observation to estimate the scale. We
        # include an error floor so that in the case of very high
        # signal-to-noise observations we pick the maximum flux value.
        signal_to_noises = np.abs(fluxes) / np.sqrt(
            flux_errors ** 2 + (1e-2 * np.max(fluxes)) ** 2
        )
        scale = np.abs(fluxes[signal_to_noises.idxmax()])

        kernel = (0.5 * scale) ** 2 * kernels.Matern32Kernel(
            [guess_length_scale ** 2, 6000 ** 2], ndim=2
        )

        if fix_scale:
            kernel.freeze_parameter("k1:log_constant")
        kernel.freeze_parameter("k2:metric:log_M_1_1")

        gp = george.GP(kernel)

        guess_parameters = gp.get_parameter_vector()

        if verbose:
            print(kernel.get_parameter_dict())

        x_data = np.vstack([times, wavelengths]).T

        gp.compute(x_data, flux_errors)

        def neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.log_likelihood(fluxes)

        def grad_neg_ln_like(p):
            gp.set_parameter_vector(p)
            return -gp.grad_log_likelihood(fluxes)

        bounds = [(0, np.log(1000 ** 2))]
        if not fix_scale:
            bounds = [(guess_parameters[0] - 10, guess_parameters[0] + 10)] + bounds

        fit_result = minimize(
            neg_ln_like, gp.get_parameter_vector(), jac=grad_neg_ln_like, bounds=bounds
        )

        if fit_result.success:
            gp.set_parameter_vector(fit_result.x)
        else:
            # Fit failed. Print out a warning, and use the initial guesses for
            # fit parameters. This only really seems to happen for objects
            # where the lightcurve is almost entirely noise.
            logger.warn(
                "GP fit failed for %s! Using guessed GP parameters. "
                "This is usually OK." % self
            )
            gp.set_parameter_vector(guess_parameters)

        if verbose:
            print(fit_result)
            print(kernel.get_parameter_dict())

        # Return the Gaussian process and associated data.
        gaussian_process = partial(gp.predict, fluxes)

        return gaussian_process, gp_observations, fit_result.x

    def get_default_gaussian_process(self):
        """Get the default Gaussian Process.

        This method calls fit_gaussian_process with the default arguments and
        caches its output so that multiple calls only require fitting the GP a
        single time.
        """
        if self._default_gaussian_process is None:
            gaussian_process, _, _ = self.fit_gaussian_process()
            self._default_gaussian_process = gaussian_process

        return self._default_gaussian_process

    def predict_gaussian_process(
        self, bands, times, uncertainties=True, fitted_gp=None, **gp_kwargs
    ):
        """Predict the Gaussian process in a given set of bands and at a given
        set of times.

        Parameters
        ==========
        bands : list(str)
            bands to predict the Gaussian process in.
        times : list or numpy.array of floats
            times to evaluate the Gaussian process at.
        uncertainties : bool (optional)
            If True (default), the GP uncertainties are computed and returned
            along with the mean prediction. If False, only the mean prediction
            is returned.
        fitted_gp : function (optional)
            By default, this function will perform the GP fit before doing
            predictions. If the GP fit has already been done, then the fitted
            GP function (returned by fit_gaussian_process) can be passed here
            instead to skip redoing the fit.
        gp_kwargs : kwargs (optional)
            Additional arguments that are passed to `fit_gaussian_process`.

        Returns
        =======
        predictions : numpy.array
            A 2-dimensional array with shape (len(bands), len(times))
            containing the Gaussian process mean flux predictions.
        prediction_uncertainties : numpy.array
            Only returned if uncertainties is True. This is an array with the
            same shape as predictions containing the Gaussian process
            uncertainty for the predictions.
        """
        if not self.executed:
            return None, None
        
        if fitted_gp is not None:
            gp = fitted_gp
        else:
            gp, _, _ = self.fit_gaussian_process(**gp_kwargs)

        # Predict the Gaussian process band-by-band.
        predictions = []
        prediction_uncertainties = []

        for band in bands:
            wavelengths = np.ones(len(times)) * get_band_central_wavelength(band)
            pred_x_data = np.vstack([times, wavelengths]).T
            if uncertainties:
                band_pred, band_pred_var = gp(pred_x_data, return_var=True)
                
                # convert back to magnitude
                band_pred += self.ref_flux[band]
                band_pred = self.convert_pred_to_mag(band_pred)
                band_pred_std = self.convert_pred_err_to_mag_err(band_pred, np.sqrt(band_pred_var))
                prediction_uncertainties.append(band_pred_std)
            else:
                band_pred = gp(pred_x_data, return_cov=False)
                band_pred += self.ref_flux[band]
                band_pred = self.convert_pred_to_mag(band_pred)

        
            
            predictions.append(band_pred)

        predictions = np.array(predictions)
        # predictions = self.convert_predictions_to_mag(predictions)

        if uncertainties:
            prediction_uncertainties = np.array(prediction_uncertainties)
            # prediction_uncertainties = self.convert_pred_err_to_mag_err(predictions, prediction_uncertainties)

            return predictions, prediction_uncertainties
        else:
            return predictions

    def plot_light_curve(self, show_gp=True, verbose=False, axis=None, **kwargs):
        """Plot the object's light curve

        Parameters
        ----------
        show_gp : bool (optional)
            If True (default), the Gaussian process prediction is plotted along
            with the raw data.
        verbose : bool (optional)
            If True, print detailed information about the light curve and GP
            fit.
        axis : `matplotlib.axes.Axes` (optional)
            The matplotlib axis to plot to. If None, a new figure will be
            created.
        kwargs : kwargs (optional)
            Additional arguments. If show_gp is True, these are passed to
            `fit_gaussian_process`. Otherwise, these are passed to
            `preprocess_observations`.
        """
        if not self.executed:
            return None, None

        if show_gp:
            gp, observations, gp_fit_parameters = self.fit_gaussian_process(
                verbose=verbose, **kwargs
            )
        else:
            observations = self.preprocess_observations(**kwargs)

        # Figure out the times to plot. We go 10% past the edges of the
        # observations.
        min_time_obs = np.min(observations["time"])
        max_time_obs = np.max(observations["time"])
        border = 0.1 * (max_time_obs - min_time_obs)
        min_time = min_time_obs - border
        max_time = max_time_obs + border

        if show_gp:
            pred_times = np.arange(min_time, max_time + 1)

            predictions, prediction_uncertainties = self.predict_gaussian_process(
                self.bands, pred_times, fitted_gp=gp
            )

        if axis is None:
            figure, axis = plt.subplots()

        for band_idx, band in enumerate(self.bands):
            mask = observations["band"] == band
            band_data = observations[mask]
            color = get_band_plot_color(band)
            marker = get_band_plot_marker(band)

            axis.errorbar(
                band_data["time"],
                band_data["mag"],
                band_data["mag_err"],
                fmt="o",
                c=color,
                markersize=6,
                marker=marker,
                label=band,
            )

            if not show_gp:
                continue

            pred = predictions[band_idx]
            axis.plot(pred_times, pred, c=color)
            err = prediction_uncertainties[band_idx]


            if kwargs.get("uncertainties", True):
                # If they were calculated, show uncertainties with a shaded
                # band.
                axis.fill_between(
                    pred_times, pred - err, pred + err, alpha=0.2, color=color
                )

        axis.legend()
        axis.set_title(self.object_id)
        axis.set_xlabel("Time (days)")
        axis.set_ylabel("Magnitude")
        axis.invert_yaxis()
        axis.set_xlim(min_time, max_time)

        axis.figure.tight_layout()



# import numpy as np

def test_gp_parameters(astro_obj, length_scales, f1_values):
    """
    Test different GP kernel parameters on an AstronomicalObject.

    Parameters
    ----------
    astro_obj : AstronomicalObject
        The astronomical object on which to perform the test.
    length_scales : list of floats
        Guess length scales (in days) to try.
    f1_values : list of floats
        Multiplicative factors for the kernel amplitude.

    Returns
    -------
    results : list of dicts
        Each entry contains the parameters tested and their chi-squared score.
    """
    results = []
    best_score = np.inf
    best_params = None

    for guess_length_scale in length_scales:
        for f1 in f1_values:
            print(f"\nTesting guess_length_scale={guess_length_scale}, f1={f1}")
            # Modify astro_obj's fit_gaussian_process call to use current parameters:
            gp, obs, params = astro_obj.fit_gaussian_process(
                verbose=False,
                guess_length_scale=guess_length_scale,
                f1=f1,
            )

            if gp is None or obs is None or len(obs) == 0:
                print("GP fitting failed or no valid observations. Skipping.")
                continue

            pred, _ = astro_obj.predict_gaussian_process(
                astro_obj.bands,
                obs["time"].values,
                fitted_gp=gp,
                uncertainties=False,
            )

            # Flatten predictions to match observations
            pred = pred.flatten()
            mags = obs["mag"].values
            mag_errs = obs["mag_err"].values

            # Compute reduced chi-squared:
            chi2 = np.sum(((mags - pred) / mag_errs) ** 2)
            dof = len(mags) - len(params)
            red_chi2 = chi2 / dof if dof > 0 else np.inf

            print(f"Reduced chi-squared: {red_chi2:.3f}")

            results.append({
                "guess_length_scale": guess_length_scale,
                "f1": f1,
                "reduced_chi2": red_chi2,
            })

            if red_chi2 < best_score:
                best_score = red_chi2
                best_params = (guess_length_scale, f1)

    print("\n=== Best parameters ===")
    print(f"guess_length_scale={best_params[0]}, f1={best_params[1]} with reduced chi2={best_score:.3f}")
    return results

# Example of how to call it:
# results = test_gp_parameters(astro_obj, length_scales=[10, 30, 50, 100], f1_values=[0.1, 0.2, 0.5])
