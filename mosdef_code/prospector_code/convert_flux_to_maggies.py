# run convert_flux_to_maggies.py
# '/Users/brianlorenz/mosdef/composite_sed_csvs/0_sed.csv'

import os
from tokenize import group
from astropy.io import ascii
from astropy.utils import data
from numpy.lib.function_base import median
from scipy import interpolate

# Savio
# median_zs_file = '/global/scratch/users/brianlorenz/median_zs.csv'

# Local
import initialize_mosdef_dirs as imd 
median_zs_file = imd.composite_seds_dir + '/median_zs.csv'

def redshift_f_lambda(data_df, groupID):
    """Since this script is also run on prospector, we only import imd here
    
    """
    zs_df = ascii.read(median_zs_file).to_pandas()
    median_z = zs_df[zs_df['groupID'] == groupID]['median_z'].iloc[0]
    data_df['f_lambda_scaled_red'] = data_df['f_lambda_scaled'] / (1+median_z)
    data_df['err_f_lambda_u_scaled_red'] = data_df['err_f_lambda_u_scaled'] / (1+median_z)
    data_df['err_f_lambda_d_scaled_red'] = data_df['err_f_lambda_d_scaled'] / (1+median_z)
    data_df['f_lambda_red'] = data_df['f_lambda'] / (1+median_z)
    data_df['err_f_lambda_u_red'] = data_df['err_f_lambda_u'] / (1+median_z)
    data_df['err_f_lambda_d_red'] = data_df['err_f_lambda_d'] / (1+median_z)
    data_df['redshifted_wavelength'] = data_df['rest_wavelength'] * (1+median_z)
    return data_df


def convert_flux_to_maggies(target_file):
    """Adds a new column in a file that converts the flux to maggies, the unit needed by Prospector

    Parameters:
    target_file (str) - location of file containing composite SED points

    """
    groupID = int(target_file.split('/')[-1][:-8])
    print(groupID)

    data_df = ascii.read(target_file).to_pandas()

    data_df = normalize_flux_5000(data_df)
    data_df = redshift_f_lambda(data_df, groupID)

    f_nu = data_df['f_lambda_red'] * (data_df['redshifted_wavelength']**2) * 3.34 * 10**(-19)
    f_jy = f_nu * (10**23)
    maggies = f_jy / 3631
    erru = data_df['err_f_lambda_u_red'] * \
        ((data_df['rest_wavelength']**2) * 3.34 * 10**(-19)) * (10**23) / 3631
    errd = data_df['err_f_lambda_d_red'] * \
        ((data_df['rest_wavelength']**2) * 3.34 * 10**(-19)) * (10**23) / 3631
    data_df['f_maggies_red'] = maggies
    data_df['err_f_maggies_u_red'] = erru
    data_df['err_f_maggies_d_red'] = errd
    data_df['err_f_maggies_avg_red'] = (erru + errd) / 2

    data_df.to_csv(target_file, index=False)
    return

def convert_all_seds_to_maggies():
    imd.check_and_make_dir(imd.mosdef_dir+'/seds_maggies')
    seds = os.listdir(imd.sed_csvs_dir)
    sed_csvs = [sedname for sedname in seds if '.csv' in sedname]
    for sed_file_name in sed_csvs:
        convert_sed_flux_to_maggies(sed_file_name)


def convert_sed_flux_to_maggies(sed_file_name):
    """Adds a new column in a file that converts the flux to maggies, the unit needed by Prospector

    Parameters:
    target_file (str) - location of file containing composite SED points

    """
    data_df = ascii.read(imd.sed_csvs_dir + '/' + sed_file_name).to_pandas()
    
    data_df['redshifted_peak_wavelength'] = data_df['peak_wavelength']*(1+data_df['Z_MOSFIRE'])
    data_df['redshifted_flux'] = data_df['f_lambda'] / (1+data_df['Z_MOSFIRE'])
    data_df['err_redshifted_flux'] = data_df['err_f_lambda'] / (1+data_df['Z_MOSFIRE'])
    f_nu = data_df['redshifted_flux'] * (data_df['redshifted_peak_wavelength']**2) * 3.34 * 10**(-19)
    f_jy = f_nu * (10**23)
    maggies = f_jy / 3631
    err = data_df['err_redshifted_flux'] * \
        ((data_df['redshifted_peak_wavelength']**2) * 3.34 * 10**(-19)) * (10**23) / 3631
    data_df['f_maggies_red'] = maggies
    data_df['err_f_maggies_red'] = err
    

    data_df.to_csv(imd.mosdef_dir+f'/seds_maggies/{sed_file_name}', index=False)
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



# convert_all_seds_to_maggies()