# Fits the emission lines of the prospector data in the same way that we fit our own data


import numpy as np
import pandas as pd
from astropy.io import ascii
import initialize_mosdef_dirs as imd
from fit_emission import fit_emission
from prospector_plot import load_obj

def setup_prospector_fit_csv(groupID, run_name):
    '''Takes the csv outputs from prospector and merges them into a better format for the fitting code
    
    Parameters:
    groupID (int): ID of the composite group to run on
    run_name (str): Name of the prospector run on Savio
    
    '''
    obs = load_obj(f'{groupID}_obs', run_name)
    # redshift_cor = (1+obs['z'])
    redshift_cor = 1 # hopefully fixed this so that a redshift correction is no longer needed in this code

    spec_df = ascii.read(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs/{groupID}_spec.csv').to_pandas()
    spec_df = spec_df.rename(columns = {'spec50_flambda' : 'f_lambda'})
    spec_df['err_f_lambda_d'] = spec_df['f_lambda'] - spec_df['spec16_flambda']
    spec_df['err_f_lambda_u'] = spec_df['spec84_flambda'] - spec_df['f_lambda']
    spec_df['f_lambda'] = spec_df['f_lambda'] * redshift_cor
    spec_df['err_f_lambda_u'] = spec_df['err_f_lambda_u'] * redshift_cor
    spec_df['err_f_lambda_d'] = spec_df['err_f_lambda_d'] * redshift_cor
    spec_df_merge = spec_df[['rest_wavelength', 'f_lambda', 'err_f_lambda_d', 'err_f_lambda_u']]

    cont_df = ascii.read(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs/{groupID}_cont_spec.csv').to_pandas()
    cont_df = cont_df.rename(columns = {'spec50_flambda' : 'cont_f_lambda'})
    cont_df['cont_err_f_lambda_d'] = cont_df['cont_f_lambda'] - cont_df['spec16_flambda']
    cont_df['cont_err_f_lambda_u'] = cont_df['spec84_flambda'] - cont_df['cont_f_lambda']
    cont_df['cont_f_lambda'] = cont_df['cont_f_lambda'] * redshift_cor
    cont_df['cont_err_f_lambda_u'] = cont_df['cont_err_f_lambda_u'] * redshift_cor
    cont_df['cont_err_f_lambda_d'] = cont_df['cont_err_f_lambda_d'] * redshift_cor
    cont_df_merge = cont_df[['rest_wavelength', 'cont_f_lambda', 'cont_err_f_lambda_d', 'cont_err_f_lambda_u']]

    spec_merged = spec_df_merge.merge(cont_df_merge, on=['rest_wavelength'])
    spec_merged.to_csv(imd.prospector_fit_csvs_dir + f'/{run_name}_csvs/{groupID}_merged_spec.csv', index=False )


def setup_all_prospector_fit_csvs(n_clusters, run_name):
    '''Runs setup_fit_csv on all of the clusters

    Parameters:
    n_clusters (int): Number of clusters
    run_name (str): Name of the prospector run on Savio
    
    '''
    for groupID in range(n_clusters):
        try:
            setup_prospector_fit_csv(groupID, run_name)
        except:
            pass


def fit_all_prospector_emission(n_clusters, run_name):
    '''Rns the fitting for all of the clusters

    Parameters:
    n_clusters (int): Number of clusters
    run_name (str): Name of the prospector run on Savio
    
    '''
    for groupID in range(n_clusters):
        try:
            fit_emission(groupID, 'cluster_norm', run_name = run_name)
        except:
            pass
    
setup_all_prospector_fit_csvs(29, 'redshift_maggies')
fit_all_prospector_emission(29, 'redshift_maggies')