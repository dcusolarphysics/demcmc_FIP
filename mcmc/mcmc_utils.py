from demcmc import (
    EmissionLine,
    TempBins,
    load_cont_funcs,
    plot_emission_loci,
    predict_dem_emcee,
    ContFuncDiscrete,
)
import numpy as np



def calc_chi2(mcmc_lines: list[EmissionLine], dem_result: np.array, temp_bins: TempBins) -> float:
    # Calculate the chi-square value for the given MCMC lines, DEM result, and temperature bins
    int_obs = np.array([line.intensity_obs for line in mcmc_lines])
    int_pred = np.array([line._I_pred(temp_bins, dem_result) for line in mcmc_lines])
    sigma_intensity_obs = np.array([line.sigma_intensity_obs for line in mcmc_lines])
    chi2 = np.sum(((int_pred - int_obs) / sigma_intensity_obs) ** 2)
    return chi2

def mcmc_process(mcmc_lines: list[EmissionLine], temp_bins: TempBins) -> np.ndarray:
    # Perform MCMC process for the given MCMC lines and temperature bins
    dem_result = predict_dem_emcee(mcmc_lines, temp_bins, nwalkers=200, nsteps=300, progress=False, dem_guess=None)
    dem_median = np.median([sample.values.value for num, sample in enumerate(dem_result.iter_binned_dems())], axis=0)
    for nstep in [300, 500]:
        dem_result = predict_dem_emcee(mcmc_lines, temp_bins, nwalkers=200, nsteps=nstep, progress=False,
                                        dem_guess=dem_median)
        dem_median = np.median([sample.values.value for num, sample in enumerate(dem_result.iter_binned_dems())],
                                axis=0)

    return dem_median
