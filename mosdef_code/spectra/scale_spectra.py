# Scales the spectra by the same amount that the composite seds were scaled

from astropy.io import ascii
import pandas as pd
import numpy as np
import initialize_mosdef_dirs as imd
from fit_emission import fit_emission

def scale_spec_to_median_halpha(groupID, norm_method, bootstrap_num=-1):
    """Scales the composite spectrum such that the flux of the halpha line is the same as the median flux of halpha for all contributing galaxies
    
    Parameters:
    bootstrap (float): Set to a number to use that spectrum. Otherwise, -1 does the composite
    """
    if bootstrap_num == -1:
        spec_df = ascii.read(imd.composite_spec_dir + f'/{norm_method}_csvs/{groupID}_spectrum.csv').to_pandas()
        emission_df = ascii.read(imd.emission_fit_csvs_dir + f'/{groupID}_emission_fits.csv').to_pandas()
    else:
        spec_df = ascii.read(imd.composite_spec_dir + f'/{norm_method}_boot_csvs/{groupID}_spectrum_{bootstrap_num}.csv').to_pandas()
        emission_df = ascii.read(imd.emission_fit_dir + f'/emission_fitting_boot_csvs/{groupID}_emission_fits_{bootstrap_num}.csv').to_pandas()
    group_df = ascii.read(imd.cluster_indiv_dfs_dir + f'/{groupID}_cluster_df.csv')

    halpha_composite = emission_df[emission_df['line_name'] == 'Halpha'].iloc[0]['flux']
    median_halpha_individuals = np.median(group_df[group_df['ha_flux']>0]['ha_flux'])
    flux_ratio = median_halpha_individuals / halpha_composite
    print(f'Individual/Spectrum halpha: {flux_ratio}')
    spec_df['f_lambda'] = spec_df['f_lambda']*flux_ratio
    spec_df['err_f_lambda'] = spec_df['err_f_lambda']*flux_ratio
    spec_df['cont_f_lambda'] = spec_df['cont_f_lambda']*flux_ratio
    if bootstrap_num == -1:
        imd.check_and_make_dir(imd.composite_spec_dir +f'/halpha_scaled_{norm_method}_csvs/')
        spec_df.to_csv(imd.composite_spec_dir +f'/halpha_scaled_{norm_method}_csvs/{groupID}_spectrum.csv')
    else:
        imd.check_and_make_dir(imd.composite_spec_dir +f'/halpha_scaled_{norm_method}_boot_csvs/')
        spec_df.to_csv(imd.composite_spec_dir +f'/halpha_scaled_{norm_method}_boot_csvs/{groupID}_spectrum_{bootstrap_num}.csv')

def scale_all_spec_to_median_halpha(n_clusters, bootstrap=-1):
    '''Runs scale_spectrum on all of the composite sed groups
    
    Parameters:
    n_clusters (int): Number of composite sed groups
    bootstrap (int): Set to number of bootstrap trials. -1 if not bootstrapping
    '''
    for groupID in range(n_clusters):
        scale_spec_to_median_halpha(groupID, 'cluster_norm')
        if bootstrap > -1:
            for bootstrap_num in range(bootstrap):
                scale_spec_to_median_halpha(groupID, 'cluster_norm', bootstrap_num=bootstrap_num)

def scale_spectrum(groupID, norm_method):
    '''Using the scaling on the photometry from convert_flux_to_maggies.py, scale the spectra and re-fit the emission

    Parameters:
    groupID (int): ID of the cluster to do the scaling on
    
    '''
    phot_df = ascii.read(imd.composite_sed_csvs_dir + f'/{groupID}_sed.csv').to_pandas()
    spec_df = ascii.read(imd.composite_spec_dir + f'/{norm_method}_csvs/{groupID}_spectrum.csv').to_pandas()

    scale = phot_df['f_lambda_scaled'].iloc[0] / phot_df['f_lambda'].iloc[0]
    spec_df['f_lambda_scaled'] = spec_df['f_lambda'] * scale
    spec_df['err_f_lambda_scaled'] = spec_df['err_f_lambda'] * scale
    spec_df['cont_f_lambda_scaled'] = spec_df['cont_f_lambda'] * scale
    spec_df.to_csv(imd.composite_spec_dir + f'/{norm_method}_csvs/{groupID}_spectrum_scaled.csv', index=False)

    fit_emission(groupID, norm_method, scaled='True')



def scale_all_spectra(n_clusters):
    '''Runs scale_spectrum on all of the composite sed groups
    
    Parameters:
    n_clusters (int): Number of composite sed groups
    '''
    for groupID in range(n_clusters):
        scale_spectrum(groupID, 'cluster_norm')

# scale_all_spectra(29)
scale_all_spec_to_median_halpha(19, bootstrap=1000)
