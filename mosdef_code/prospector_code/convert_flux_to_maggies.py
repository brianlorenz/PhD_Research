# run convert_flux_to_maggies.py
# '/Users/brianlorenz/mosdef/composite_sed_csvs/0_sed.csv'

import os
import numpy as np
import pandas as pd
from astropy.io import ascii
from read_data import mosdef_df
from mosdef_obj_data_funcs import get_mosdef_obj, read_sed
import initialize_mosdef_dirs as imd
from scipy import interpolate



def convert_flux_to_maggies(target_file):
    """Adds a new column in a file that converts the flux to maggies, the unit needed by Prospector

    Parameters:
    target_file (str) - location of file containing composite SED points

    """
    data = ascii.read(target_file).to_pandas()

    data = normalize_flux_5000(data)

    f_nu = data['f_lambda_scaled'] * (data['rest_wavelength']**2) * 3.34 * 10**(-19)
    f_jy = f_nu * (10**23)
    maggies = f_jy / 3631
    erru = data['err_f_lambda_u_scaled'] * \
        ((data['rest_wavelength']**2) * 3.34 * 10**(-19)) * (10**23) / 3631
    errd = data['err_f_lambda_d_scaled'] * \
        ((data['rest_wavelength']**2) * 3.34 * 10**(-19)) * (10**23) / 3631
    data['f_maggies'] = maggies
    data['err_f_maggies_u'] = erru
    data['err_f_maggies_d'] = errd
    data['err_f_maggies_avg'] = (erru + errd) / 2

    data.to_csv(target_file, index=False)
    return


def normalize_flux_5000(sed_df):
    """Normalizes the sed points so that they are scaled to have a flux of 10^-16 at 5000 angstroms

    Parameters:
    sed_df (pd.DataFrame): Pandas dataframe containing the composite sed
    """
    sed_interp = interpolate.interp1d(sed_df['rest_wavelength'], sed_df['f_lambda'])
    scale = 10**(-16) / sed_interp(5000)
    sed_df['f_lambda_scaled'] = sed_df['f_lambda']*scale
    sed_df['err_f_lambda_u_scaled'] = sed_df['err_f_lambda_u']*scale
    sed_df['err_f_lambda_d_scaled'] = sed_df['err_f_lambda_d']*scale
    return sed_df
    


def convert_folder_to_maggies(target_folder):
    """Adds a new column to all files in a folder that converts the flux to maggies, the unit needed by Prospector

    Parameters:
    target_folder (str) - location of folder containing composite SED csvs

    """

    # Figure out which files  in the folder are csv filters:
    data_files = [file for file in os.listdir(target_folder) if '.csv' in file]

    for data_file in data_files:
        convert_flux_to_maggies(target_folder + '/' + data_file)
    return


def prospector_maggies_to_flux(obs, phot):
    """Converts the outputs from prospector back into f_lambda units

    Parameters:
    obs (prospector): obs dictionary
    phot (array, prospector): Array of model photometry in maggies

    Returns:
    obs (prospector): obs dictionary
    f_lambda_res (array): Array of model photometry in f_lambda units


    """
    f_jy_obs = obs['maggies'] * 3631
    f_nu_obs = f_jy_obs / (10**23)
    f_lambda_obs = f_nu_obs / ((obs['phot_wave']**2) * 3.34 * 10**(-19))
    obs['f_lambda'] = f_lambda_obs

    err_f_jy_obs = obs['maggies_unc'] * 3631
    err_f_nu_obs = err_f_jy_obs / (10**23)
    err_f_lambda_obs = err_f_nu_obs / \
        ((obs['phot_wave']**2) * 3.34 * 10**(-19))
    obs['err_f_lambda'] = err_f_lambda_obs

    f_jy_res = phot * 3631
    f_nu_res = f_jy_res / (10**23)
    f_lambda_res = f_nu_res / ((obs['phot_wave']**2) * 3.34 * 10**(-19))

    return obs, f_lambda_res


def prospector_maggies_to_flux_spec(spec_wave, spec):
    """Converts the outputs from prospector back into f_lambda units

    Parameters:
    spec_wave (array): spectrum wavelength
    spec (array, prospector): spectrum

    Returns:
    f_lambda_spec (array): Spectrum converted to f_lambda


    """
    f_jy_spec = spec * 3631
    f_nu_spec = f_jy_spec / (10**23)
    f_lambda_spec = f_nu_spec / ((spec_wave**2) * 3.34 * 10**(-19))

    return f_lambda_spec




