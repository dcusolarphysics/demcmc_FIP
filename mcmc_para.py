from multiprocessing import Pool
from functools import partial
from tqdm import tqdm
import numpy as np
import astropy.units as u
from ashmcmc import ashmcmc, interp_emis_temp

from demcmc import (
    EmissionLine,
    TempBins,
    load_cont_funcs,
    plot_emission_loci,
    predict_dem_emcee,
    ContFuncDiscrete,
)
from demcmc.units import u_temp, u_dem

def calc_chi2(mcmc_lines: list[EmissionLine], dem_result: np.array, temp_bins: TempBins) -> float:
    # Calculate the chi-square value for the given MCMC lines, DEM result, and temperature bins
    int_obs = np.array([line.intensity_obs for line in mcmc_lines])
    int_pred = np.array([line._I_pred(temp_bins, dem_result) for line in mcmc_lines])
    sigma_intensity_obs = np.array([line.sigma_intensity_obs for line in mcmc_lines])

    chi2_line = ((int_pred - int_obs) / sigma_intensity_obs) ** 2
    chi2 = np.sum(chi2_line)
    return chi2

def mcmc_process(mcmc_lines: list[EmissionLine], temp_bins: TempBins) -> np.ndarray:
    # Perform MCMC process for the given MCMC lines and temperature bins
    dem_result = predict_dem_emcee(mcmc_lines, temp_bins, nwalkers=200, nsteps=300, progress=False, dem_guess=None)
    dem_init = np.median([sample.values.value for num, sample in enumerate(dem_result.iter_binned_dems())], axis=0)
    dem_result = predict_dem_emcee(mcmc_lines, temp_bins, nwalkers=200, nsteps=3000, progress=False,
                                    dem_guess=dem_init)
    dem_median = np.median([sample.values.value for num, sample in enumerate(dem_result.iter_binned_dems())],
                            axis=0)

    return dem_median

def check_dem_exists(filename: str) -> bool:
    # Check if the DEM file exists
    from os.path import exists
    return exists(filename)    

def process_pixel(args: tuple[int, np.ndarray, np.ndarray, list[str], np.ndarray, ashmcmc]) -> None:
    from pathlib import Path
    # Process a single pixel with the given arguments
    xpix, Intensity, Int_error, Lines, ldens, a = args
    output_file = f'{a.outdir}/dem_columns/dem_{xpix}.npz'
    # Extract the directory path from the output_file
    output_dir = Path(output_file).parent

    # Check if the directory exists, and create it if it doesn't
    output_dir.mkdir(parents=True, exist_ok=True)

    ycoords_out = []
    dem_results = []
    chi2_results = []
    linenames_list = []

    if not check_dem_exists(output_file):
        for ypix in tqdm(range(Intensity.shape[0])):

            logt, emis, linenames = a.read_emissivity(ldens[ypix, xpix])
            logt_interp = interp_emis_temp(logt.value)
            temp_bins = TempBins(logt_interp * u.K)
            # loc = np.where((np.log10(logt_interp) >= 4) & (np.log10(logt_interp) <= 8))
            emis_sorted = a.emis_filter(emis, linenames, Lines)
            mcmc_lines = []

            for ind, line in enumerate(Lines):
                if (line[:2] == 'fe') and (Intensity[ypix, xpix, ind] > 10):
                    mcmc_emis = emis_sorted[ind, :]
                    mcmc_emis = ContFuncDiscrete(logt_interp*u.K, interp_emis_temp(emis_sorted[ind, :]) * u.cm ** 5 / u.K,
                                                name=line)
                    mcmc_intensity = Intensity[ypix, xpix, ind]
                    mcmc_int_error = max(Int_error[ypix, xpix, ind], 0.25 * mcmc_intensity)
                    emissionLine = EmissionLine(
                        mcmc_emis,
                        intensity_obs=mcmc_intensity,
                        sigma_intensity_obs=mcmc_int_error,
                        name=line
                    )
                    mcmc_lines.append(emissionLine)

            dem_median = mcmc_process(mcmc_lines, temp_bins) # Run 2 MCMC processes and return the median DEM
            chi2 = calc_chi2(mcmc_lines, dem_median, temp_bins)
            dem_results.append(dem_median)
            chi2_results.append(chi2)
            ycoords_out.append(ypix)
            linenames_list.append(mcmc_lines)



        dem_results = np.array(dem_results)
        chi2_results = np.array(chi2_results)
        linenames_list = np.array(linenames_list, dtype=object)

        np.savez(output_file, dem_results=dem_results, chi2=chi2_results, ycoords_out=ycoords_out, lines_used=linenames_list, logt = np.array(logt_interp))

def download_data(filename: str) -> None:
    from eispac.download import download_hdf5_data
    download_hdf5_data(filename, local_top='SO_EIS_data', overwrite=False)

def combine_dem_files(xdim:int, ydim:int, dir: str) -> np.array:
    from glob import glob
    from re import search

    dem_files = sorted(glob(f'{dir}/dem_columns/dem*.npz'))
    ref = np.load(dem_files[0])['dem_results'].shape
    logt = np.load(dem_files[0])['logt']
    dem_combined = np.zeros((ydim,xdim,ref[1]))
    chi2_combined = np.zeros((ydim,xdim))
    lines_used = np.zeros((ydim,xdim))

    for dem_file in dem_files:
        xpix_loc = search(r'dem_(\d+)\.npz$', dem_file).group(1)
        dem_combined[:,int(xpix_loc), :] = np.load(dem_file)['dem_results'] 
        chi2_combined[:,int(xpix_loc)] = np.load(dem_file)['chi2'] 
        lines_used[:,int(xpix_loc)] = np.array([len(line) for line in np.load(dem_file, allow_pickle=True)['lines_used']])
    return dem_combined, chi2_combined, lines_used, logt

def process_data(filename: str) -> None:
    # Create an ashmcmc object with the specified filename
    import platform
    download_data(filename)
    a = ashmcmc(filename)

    # Retrieve necessary data from ashmcmc object
    Lines, Intensity, Int_error = a.fit_data(plot=False)
    ldens = a.read_density()
    
    # Generate a list of arguments for process_pixel function
    args_list = [(xpix, Intensity, Int_error, Lines, ldens, a) for xpix in range(Intensity.shape[1])]

    # Determine the operating system type (Linux or macOS)
    # Set the number of processes based on the operating system
    if platform.system() == "Linux": process_num = 70
    elif platform.system() == "Darwin": process_num = 10
    else: process_num = 10 

    # Create a Pool of processes for parallel execution
    with Pool(processes=process_num) as pool:
        results = list(tqdm(pool.imap(process_pixel, args_list), total=len(args_list), desc="Processing Pixels"))

    # Combine the DEM files into a single array
    print('------------------------------Combining DEM files------------------------------')
    dem_combined, chi2_combined, lines_used, logt = combine_dem_files(Intensity.shape[1], Intensity.shape[0], a.outdir)
    np.savez(f'{a.outdir}/{a.outdir}_dem_combined.npz', dem_combined=dem_combined, chi2_combined=chi2_combined, lines_used=lines_used, logt=logt)
    return f'{a.outdir}/{a.outdir}_dem_combined.npz'


def pred_intensity_compact(emis:np.array, logt:np.array, linename:str, dem:np.array) -> float:
    mcmc_emis = ContFuncDiscrete(logt*u.K, interp_emis_temp(emis) * u.cm ** 5 / u.K,
                                name=linename)
    emissionLine = EmissionLine(
        mcmc_emis,
        name=linename
    )
    temp_bins = TempBins(logt * u.K)
    return emissionLine._I_pred(temp_bins, dem)

def correct_metadata(map, ratio_name):
    # Correct the metadata of the map
    map.meta['measrmnt'] = 'FIP Bias'
    map.meta.pop('bunit', None)
    map.meta['line_id'] = ratio_name
    return map

def calc_composition(filename, np_file, line_database):
    # I am tired and am probably very dumb in calculating this
    from sunpy.map import Map
    a = ashmcmc(filename)

    ldens = a.read_density()
    dem_median = np.load(np_file)['dem_combined']

    # Retrieve necessary data from ashmcmc object
    for comp_ratio in line_databases:
        intensities = np.zeros((ldens.shape[0], ldens.shape[1], 2))
        composition = np.zeros_like(ldens)  # Initialize composition array

        for num, fip_line in enumerate(line_databases[comp_ratio][:2]):  # Iterate only over the first 2 lines
            map = a.ash.get_intensity(fip_line, outdir=a.outdir, plot=False)
            intensities[:, :, num] = map.data

        for ypix, xpix in np.ndindex(ldens.shape):  # Iterate over each pixel
            logt, emis, linenames = a.read_emissivity(ldens[ypix, xpix]) # Read emissivity from .sav files
            logt_interp = interp_emis_temp(logt.value) # Interpolate the temperature
            temp_bins = TempBins(logt_interp * u.K) # Create temp_bin structure for intensity prediction
            emis_sorted = a.emis_filter(emis, linenames, line_databases[comp_ratio][:2]) # Filter emissivity based on specified lines

            int_lf = pred_intensity_compact(emis_sorted[0], logt_interp, line_databases[comp_ratio][0], dem_median)
            dem_scaled = dem_median * (intensities[ypix, xpix, 0] / int_lf)
            int_hf = pred_intensity_compact(emis_sorted[1], logt_interp, line_databases[comp_ratio][1], dem_scaled)
            fip_ratio = int_hf/intensities[ypix, xpix, 1]
            composition[ypix, xpix] = fip_ratio  # Update composition matrix

        # Create SunPy Map with appropriate metadata
        map_fip = Map(composition, map.meta)
        map_fip = correct_metadata(map_fip, comp_ratio[2])
        map_fip.save(f'{a.outdir}/{a.outdir}_{comp_ratio[2]}.fits')



if __name__ == "__main__":
    # filename = 'SO_EIS_data/eis_20230405_220513.data.h5'
    filenames = ['SO_EIS_data/eis_20230327_061218.data.h5',
    'SO_EIS_data/eis_20230327_074942.data.h5',
    'SO_EIS_data/eis_20230327_092942.data.h5',
    'SO_EIS_data/eis_20230327_112937.data.h5',
    'SO_EIS_data/eis_20230327_121141.data.h5',
    'SO_EIS_data/eis_20230327_131642.data.h5',
    'SO_EIS_data/eis_20230327_143341.data.h5',
    'SO_EIS_data/eis_20230327_163141.data.h5',
    'SO_EIS_data/eis_20230327_180811.data.h5',
    'SO_EIS_data/eis_20230327_194441.data.h5',
    'SO_EIS_data/eis_20230327_212141.data.h5',
    'SO_EIS_data/eis_20230327_225811.data.h5',
    'SO_EIS_data/eis_20230328_002912.data.h5',
    'SO_EIS_data/eis_20230328_015542.data.h5',
    'SO_EIS_data/eis_20230328_033248.data.h5',
    'SO_EIS_data/eis_20230328_050911.data.h5',
    'SO_EIS_data/eis_20230328_064711.data.h5',
    'SO_EIS_data/eis_20230328_100341.data.h5',
    'SO_EIS_data/eis_20230328_115313.data.h5',
    'SO_EIS_data/eis_20230328_125814.data.h5',
    'SO_EIS_data/eis_20230328_141513.data.h5',
    'SO_EIS_data/eis_20230328_152013.data.h5',
    'SO_EIS_data/eis_20230328_170613.data.h5',
    'SO_EIS_data/eis_20230328_184243.data.h5',
    'SO_EIS_data/eis_20230328_201913.data.h5',
    'SO_EIS_data/eis_20230328_215643.data.h5']
    for filename in filenames:
        np_file = process_data(filename)
        line_databases = {
            "sis" :['si_10_258.37','s_10_264.23', 'Si X-S X'],
            # "fear" : ['fe_14_264.79', 'ar_11_188.81', 'Fe XVI-Ar XI']
        }
        calc_composition(filename, np_file, line_databases)