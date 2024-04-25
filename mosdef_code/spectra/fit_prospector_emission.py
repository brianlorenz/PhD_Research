# Fits the emission lines of the prospector data in the same way that we fit our own data


import numpy as np
import pandas as pd
from astropy.io import ascii
import initialize_mosdef_dirs as imd
from fit_emission import fit_emission
from prospector_plot import load_obj
from cosmology_calcs import flux_to_luminosity
import os

def setup_prospector_fit_csv(groupID, run_name):
    '''Takes the csv outputs from prospector and merges them into a better format for the fitting code
    
    Parameters:
    groupID (int): ID of the composite group to run on
    run_name (str): Name of the prospector run on Savio
    
    '''
    obs = load_obj(f'group{groupID}_obs', run_name)
    # redshift_cor = (1+obs['z'])
    redshift_cor = 1 # hopefully fixed this so that a redshift correction is no longer needed in this code

    spec_df = ascii.read(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs/group{groupID}_spec.csv').to_pandas()
    spec_df = spec_df.rename(columns = {'spec50_flambda' : 'f_lambda'})
    spec_df['err_f_lambda_d'] = spec_df['f_lambda'] - spec_df['spec16_flambda']
    spec_df['err_f_lambda_u'] = spec_df['spec84_flambda'] - spec_df['f_lambda']
    spec_df['f_lambda'] = spec_df['f_lambda'] * redshift_cor
    spec_df['err_f_lambda_u'] = spec_df['err_f_lambda_u'] * redshift_cor
    spec_df['err_f_lambda_d'] = spec_df['err_f_lambda_d'] * redshift_cor
    spec_df_merge = spec_df[['rest_wavelength', 'f_lambda', 'err_f_lambda_d', 'err_f_lambda_u']]

    cont_df = ascii.read(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs/group{groupID}_cont_spec.csv').to_pandas()
    cont_df = cont_df.rename(columns = {'spec50_flambda' : 'cont_f_lambda'})
    cont_df['cont_err_f_lambda_d'] = cont_df['cont_f_lambda'] - cont_df['spec16_flambda']
    cont_df['cont_err_f_lambda_u'] = cont_df['spec84_flambda'] - cont_df['cont_f_lambda']
    cont_df['cont_f_lambda'] = cont_df['cont_f_lambda'] * redshift_cor
    cont_df['cont_err_f_lambda_u'] = cont_df['cont_err_f_lambda_u'] * redshift_cor
    cont_df['cont_err_f_lambda_d'] = cont_df['cont_err_f_lambda_d'] * redshift_cor
    cont_df_merge = cont_df[['rest_wavelength', 'cont_f_lambda', 'cont_err_f_lambda_d', 'cont_err_f_lambda_u']]

    spec_merged = spec_df_merge.merge(cont_df_merge, on=['rest_wavelength'])
    spec_merged.to_csv(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs/{groupID}_merged_spec.csv', index=False )


def setup_all_prospector_fit_csvs(n_clusters, run_name, ignore_groups=[]):
    '''Runs setup_fit_csv on all of the clusters

    Parameters:
    n_clusters (int): Number of clusters
    run_name (str): Name of the prospector run on Savio
    
    '''
    for groupID in range(n_clusters):
        if groupID in ignore_groups:
            continue
        # Check if that group had a successful prospector run
        if confirm_h5file_exists(groupID, run_name) == True:
            setup_prospector_fit_csv(groupID, run_name)
        


def fit_all_prospector_emission(n_clusters, run_name, ignore_groups=[]):
    '''Rns the fitting for all of the clusters

    Parameters:
    n_clusters (int): Number of clusters
    run_name (str): Name of the prospector run on Savio
    
    '''
    for groupID in range(n_clusters):
        if groupID in ignore_groups:
            continue
        if confirm_h5file_exists(groupID, run_name) == True:
            fit_emission(groupID, 'cluster_norm', run_name = run_name, prospector=True)
       
    
def confirm_h5file_exists(groupID, run_name):
    h5_dir = os.listdir(imd.prospector_h5_dir+f'/{run_name}_h5s/')
    groupfiles = [filename.split('_')[1] for filename in h5_dir]
    if f'group{groupID}' in groupfiles:
        exists = True
    else:
        exists = False
    return exists

def multiply_fit_by_lumdist(n_clusters, run_name, ignore_groups=[]):
    median_zs = ascii.read(imd.median_zs_file).to_pandas()
    for groupID in range(n_clusters):
        if groupID in ignore_groups:
            continue
        if confirm_h5file_exists(groupID, run_name) == True:
            redshift = median_zs.iloc[groupID]['median_z']
            emission_fit_df = ascii.read(imd.prospector_emission_fits_dir + f'/{run_name}_emission_fits/{groupID}_emission_fits.csv').to_pandas()
            fluxes = emission_fit_df['flux']
            emission_fit_df['luminosity'] = flux_to_luminosity(fluxes, redshift)
            err_fluxes = emission_fit_df['err_flux']
            emission_fit_df['err_luminosity'] = flux_to_luminosity(err_fluxes, redshift)
            emission_fit_df.to_csv(imd.prospector_emission_fits_dir + f'/{run_name}_emission_fits/{groupID}_emission_fits.csv', index=False)
# ignore_groups = [0,5,12,19,22]
# setup_all_prospector_fit_csvs(23, 'dust_index_test', ignore_groups)
# fit_all_prospector_emission(23, 'dust_index_test', ignore_groups)