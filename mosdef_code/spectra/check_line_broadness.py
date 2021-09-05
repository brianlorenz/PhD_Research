

import initialize_mosdef_dirs as imd
from astropy.io import ascii
import pandas as pd
import numpy as np
from scipy import interpolate
import scipy.integrate as integrate
from scipy.optimize import curve_fit

import os
import matplotlib.pyplot as plt



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

    # Test plot, looks good, it grabs the line
    # wave_plot = np.arange(6553, 6573, 0.2)
    # plt.plot(wave_plot, spec_interp(wave_plot))
    # plt.show()

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
        filt_interp = interpolate.interp1d(filt['rest_wavelength'], filt['transmission'], bounds_error=False)

        # Test plot, looks good, it grabs the filter
        # wave_plot = np.arange(20000, 33000, 0.73)
        # plt.plot(wave_plot, filt_interp(wave_plot))
        # plt.show()

        def flux_func_numerator(wave):
            """Function that you need to integrate to get the flux"""

            return spec_interp(wave)*filt_interp(wave)*wave*10**18

        def flux_func_denominator(wave):
            """Function that you need to integrate to get the flux"""

            return filt_interp(wave)*wave 

        # numerator = integrate.quad(flux_func_numerator, 801, 25000)[0]
        # denominator = integrate.quad(flux_func_denominator, 801, 25000)[0]
        # Testing trapz integration
        wave_array = np.arange(801, 39999, 0.1)
        numerator = integrate.trapz(flux_func_numerator(wave_array))
        denominator = integrate.trapz(flux_func_denominator(wave_array))
        flux = numerator / denominator
        print(f'Num: {numerator}')
        print(f'Dem: {denominator}')
        print(f'-----------------')
        points.append(int(point))
        fluxes.append(flux / 10**18)

    line_width_df = pd.DataFrame(zip(points, fluxes), columns=['rest_wavelength', 'flux'])
    line_width_df.to_csv(imd.line_widths_dir + f'/group{groupID}_{rest_wave}_broadness.csv', index=False)
    

def plot_broadness(groupID, rest_waves):
    '''Plots the broadness of all of the provided lines on one axis
    
    Parameters:
    groupID (int): ID of the composite
    rest_waves (list): List of peak wavelengths in angtroms, rounded
    '''
    colors = ['black','blue']
    run_count = 0
    min_bounds = []
    max_bounds = []

    fig, ax = plt.subplots(figsize = (8,8))

    for rest_wave in rest_waves:
        line_width_df = ascii.read(imd.line_width_csvs_dir + f'/group{groupID}_{rest_wave}_broadness.csv').to_pandas()
        line_width_df['flux'] = line_width_df['flux']*10**18

        guess = [6563, 3000, 50]

        def gaussian(x, mu, sig, amp):
            return amp * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        popt, pcov = curve_fit(gaussian, line_width_df['rest_wavelength'], line_width_df['flux'], p0=guess)
        mean = popt[0]
        sigma = popt[1]
        amp = popt[2]

        gauss_waves = np.arange(rest_wave-4000, rest_wave+4000, 1)
        gauss_ys = gaussian(gauss_waves, mean, sigma, amp)

        

        ax.plot(line_width_df['rest_wavelength'], line_width_df['flux'], marker='o', ls='None', color=colors[run_count])
        ax.plot(gauss_waves, gauss_ys, marker='None', ls='-', color='red')
        if run_count==0:
            ylim = ax.get_ylim()

        min_bound = mean-2*np.abs(sigma)
        max_bound = mean+2*np.abs(sigma)
        min_bounds.append(min_bound)
        max_bounds.append(max_bound)
    
        ax.plot([min_bound, min_bound], [-1000, 1000], ls='--', marker='None', color=colors[run_count])
        ax.plot([max_bound, max_bound], [-1000, 1000], ls='--', marker='None', color=colors[run_count])
        run_count += 1
    ax.set_xscale('log')
    ax.set_ylabel('Flux (*10^18)')
    ax.set_xlabel('Wavelength ($\AA$)')
    ax.set_ylim(ylim)
    fig.savefig(imd.line_width_images_dir + f'/{groupID}_widths.pdf')

    # Save the bounds
    bounds_df = pd.DataFrame(zip(rest_waves, min_bounds, max_bounds), columns=['rest_wavelength', 'min_bound', 'max_bound'])
    bounds_df.to_csv(imd.line_widths_dir + f'/{groupID}_bounds.csv', index=False)

for groupID in range(0, 29):
    try:
        plot_broadness(groupID, [6563,5007])
    except:
        pass

# for groupID in range(0, 29):
#     try:
#         check_broadness(groupID, 'redshift_maggies', 5007)
#     except:
#         pass

