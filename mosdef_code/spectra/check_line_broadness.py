

import initialize_mosdef_dirs as imd
from astropy.io import ascii
import pandas as pd
import numpy as np
from scipy import interpolate
import scipy.integrate as integrate
import os

def check_broadness(groupID, run_name, rest_wave, width=15):
    """Makes a plot of how broad a line is by convolving it with all of the filters
    
    Parameters:
    groupID (int): ID of the group to convolve
    run_name (str): Name of the prospector run that you are looking at to convolve
    rest_wave (int): Wavelength closest to the line
    width (int): Angstroms on either side of the line to consider in the convolution
    """

    # Read in the spectrum
    spec_df = ascii.read(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs/{groupID}_spec.csv').to_pandas()
    spec_df_cut = spec_df[np.logical_and(spec_df['rest_wavelength']>rest_wave-width, spec_df['rest_wavelength']<rest_wave+1+width)]
    spec_interp = interpolate.interp1d(spec_df_cut['rest_wavelength'], spec_df_cut['spec50_flambda'], bounds_error=False, fill_value=0)

    # Find the filters
    filt_folder = imd.composite_filter_csvs_dir + f'/{groupID}_filter_csvs/'
    filt_files = [file for file in os.listdir(filt_folder) if '.csv' in file]

    # loop over each point, storing both the point and the integrated flux value at that point
    points = []
    fluxes = []
    for i in range(len(filt_files)):
        filt_file = filt_files[i]
        point = filt_file.split('.')[0].split('_')[1]
        print(f'Reading in filter for point {point}...')
        filt = ascii.read(filt_folder + filt_file).to_pandas()
        filt_interp = interpolate.interp1d(filt['rest_wavelength'], filt['transmission'])

        def flux_func_numerator(wave):
            """Function that you need to integrate to get the flux"""

            return spec_interp(wave)*filt_interp(wave)*wave 

        def flux_func_denominator(wave):
            """Function that you need to integrate to get the flux"""

            return filt_interp(wave)*wave 

        flux = integrate.quad(flux_func_numerator, 801, 39999) / integrate.quad(flux_func_denominator, 801, 39999)

        points.append(int(point))
        fluxes.append(flux)



check_broadness(0, 'first_savio', 6563)

