# Scales the spectra by the same amount that the composite seds were scaled

from astropy.io import ascii
import pandas as pd
import numpy as np
import initialize_mosdef_dirs as imd
from fit_emission import fit_emission

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