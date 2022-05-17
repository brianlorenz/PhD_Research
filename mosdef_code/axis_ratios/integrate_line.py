from scipy import integrate
import numpy as np
from astropy.io import ascii
import initialize_mosdef_dirs as imd
import pandas as pd

def re_calc_emission_flux(n_groups, save_name, bootstrap=-1):
    """Recalculate the emission flux, using integration rather than Gaussian fits 
    
    n_groups (int): Number of groups
    save_name (str): folder where groups are stored
    """

    for axis_group in range(n_groups):
        
        if bootstrap > 0:
            bootstrap_count = 0
        else:
            bootstrap_count = -2
        while bootstrap_count < bootstrap:
            if bootstrap_count == -2:
                emission_df_loc = imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_fits/{axis_group}_emission_fits.csv'
                cont_sub_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_cont_subs/{axis_group}_cont_sub.csv').to_pandas()
            else:
                emission_df_loc = imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_fits_boots/{axis_group}_emission_fits_{bootstrap_count}.csv'
                cont_sub_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_cont_subs_boots/{axis_group}_cont_sub_{bootstrap_count}.csv').to_pandas()
            emission_fit_df = ascii.read(emission_df_loc).to_pandas()
            emission_fit_df['gaussian_flux'] = emission_fit_df['flux']
            emission_fit_df['err_gaussian_flux'] = emission_fit_df['err_flux']

            line_fluxes = []
            # At each line, find a 3-sigma region on either side, and integrate the flux
            for i in range(len(emission_fit_df)):
                emission_row = emission_fit_df.iloc[i]
                sigma = emission_row['sigma']
                center = emission_row['line_center_rest']
                wavelength_range = (center - 3*sigma, center + 3*sigma)
                wavelength_idxs = np.logical_and(cont_sub_df['wavelength_cut'] > wavelength_range[0], cont_sub_df['wavelength_cut'] < wavelength_range[1])
                fluxes = cont_sub_df[wavelength_idxs]['continuum_sub_ydata']
                line_flux = integrate_line(fluxes)
                line_fluxes.append(line_flux)
            emission_fit_df['flux'] = line_fluxes

            emission_fit_df.to_csv(emission_df_loc, index=False)

            if bootstrap == -2:
                bootstrap_count = bootstrap
            else:
                bootstrap_count = bootstrap_count + 1

def integrate_line(fluxes, resolution = 0.5):
    """Find the area under the curve of a line with scipy

    fluxes (array): Continuum-subtracted fluxes, clipped to only be over the range desired
    resolution (float): Wavelength separation in Angstroms
    """
    int_res = integrate.simps(fluxes, x=None, dx=resolution, axis=-1, even='avg')
    return int_res


# re_calc_emission_flux(8, 'both_sfms_4bin_median_2axis_boot100')
re_calc_emission_flux(8, 'both_sfms_4bin_median_2axis_boot100', bootstrap=100)