# Functions to open the data tables in mosdef and store them it into
# pandas dataframes

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from tabulate import tabulate
from astropy.table import Table
import initialize_mosdef_dirs as imd


def read_file(data_file):
    """Reads the fits file into a pandas table

    Parameters:
    data_file (string): location of the file to read


    Returns:
    df (pd.DataFrame): pandas dataframe of the fits table
    """
    data = Table.read(data_file, format='fits')
    df = data.to_pandas()
    return df


mosdef_df = read_file(imd.loc_mosdef0d)
# The 'FIELD' column is in bytes format - here we convert to a string
mosdef_df['FIELD_STR'] = [mosdef_df.iloc[i]['FIELD'].decode(
    "utf-8").rstrip() for i in range(len(mosdef_df))]
linemeas_df = read_file(imd.mosdef_dir + '/Mosdef_cats/linemeas_latest.fits')
sfrs_df = read_file(imd.loc_sfrs_latest)
agnflag_df = read_file(imd.mosdef_dir + '/Mosdef_cats/agnflag_latest.fits')
metal_df = read_file(imd.mosdef_dir + '/Mosdef_cats/mosdef_metallicity_latest.fits')


def get_shapley_sample():
    redshift_idxs = np.logical_and(mosdef_df['Z_MOSFIRE']>2.09, mosdef_df['Z_MOSFIRE']<2.61)
    ha_snr = linemeas_df['HA6565_FLUX'] / linemeas_df['HA6565_FLUX_ERR']
    hb_snr = linemeas_df['HB4863_FLUX'] / linemeas_df['HB4863_FLUX_ERR']
    nii_ha_flag = np.log10(linemeas_df['NIIHA'])<-0.3
    snr_idxs = np.logical_and(ha_snr>3, hb_snr>3)
    redshift_and_snr_idxs = np.logical_and(snr_idxs, redshift_idxs)
    redshift_and_snr_and_niiha_idxs = np.logical_and(redshift_and_snr_idxs, nii_ha_flag)
    sample_df = mosdef_df[redshift_and_snr_and_niiha_idxs]
    sample_df = sample_df.reset_index()
    ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()
    merged_df = pd.merge(sample_df, ar_df, left_on=['V4ID', 'FIELD_STR'], right_on=['v4id', 'field'], how='left')
    agn_idxs = merged_df['agn_flag']==0
    ha_detected = merged_df['ha_detflag_sfr'] == 0
    hb_detected = merged_df['hb_detflag_sfr'] == 0
    merged_idxs = np.logical_and(np.logical_and(ha_detected, hb_detected), agn_idxs)
    sample_df = sample_df[merged_idxs]
    return sample_df
    

