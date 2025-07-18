from read_data import linemeas_df, mosdef_df
import initialize_mosdef_dirs as imd
import numpy as np
import pandas as pd
from astropy.io import ascii

def get_shapley_sample():
    redshift_idxs = np.logical_and(mosdef_df['Z_MOSFIRE']>2.09, mosdef_df['Z_MOSFIRE']<2.61)
    ha_snr = linemeas_df['HA6565_FLUX'] / linemeas_df['HA6565_FLUX_ERR']
    hb_snr = linemeas_df['HB4863_FLUX'] / linemeas_df['HB4863_FLUX_ERR']
    nii_ha_flag = np.log10(linemeas_df['NIIHA'])<-0.3
    snr_idxs = np.logical_and(ha_snr>3, hb_snr>3)
    redshift_and_snr_idxs = np.logical_and(snr_idxs, redshift_idxs)
    redshift_and_snr_and_niiha_idxs = np.logical_and(redshift_and_snr_idxs, nii_ha_flag)
    sample_df = mosdef_df[redshift_and_snr_and_niiha_idxs]
    linemeas_df2 = linemeas_df[redshift_and_snr_and_niiha_idxs]
    sample_df = sample_df.reset_index()
    linemeas_df2 = linemeas_df2.reset_index()
    ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()
    merged_df = pd.merge(sample_df, ar_df, left_on=['V4ID', 'FIELD_STR'], right_on=['v4id', 'field'], how='left')
    agn_idxs = merged_df['agn_flag']==0
    ha_detected = merged_df['ha_detflag_sfr'] == 0
    hb_detected = merged_df['hb_detflag_sfr'] == 0
    merged_idxs = np.logical_and(np.logical_and(ha_detected, hb_detected), agn_idxs)
    sample_df = sample_df[merged_idxs]
    linemeas_df2 = linemeas_df2[merged_idxs]
    return sample_df, linemeas_df2

def get_mosdef_compare_sample():
    # redshift_idxs = np.logical_and(mosdef_df['Z_MOSFIRE']>2.09, mosdef_df['Z_MOSFIRE']<2.61)
    redshift_idxs = mosdef_df['Z_MOSFIRE']<2.61
    ha_snr = linemeas_df['HA6565_FLUX'] / linemeas_df['HA6565_FLUX_ERR']
    hb_snr = linemeas_df['HB4863_FLUX'] / linemeas_df['HB4863_FLUX_ERR']
    nii_ha_flag = np.log10(linemeas_df['NIIHA'])<-0.3
    snr_idxs = np.logical_and(ha_snr>3, hb_snr>3)
    redshift_and_snr_idxs = np.logical_and(snr_idxs, redshift_idxs)
    redshift_and_snr_and_niiha_idxs = np.logical_and(redshift_and_snr_idxs, nii_ha_flag)
    sample_df = mosdef_df[redshift_and_snr_and_niiha_idxs]
    linemeas_df2 = linemeas_df[redshift_and_snr_and_niiha_idxs]
    sample_df = sample_df.reset_index()
    linemeas_df2 = linemeas_df2.reset_index()
    ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()
    merged_df = pd.merge(sample_df, ar_df, left_on=['V4ID', 'FIELD_STR'], right_on=['v4id', 'field'], how='left')
    agn_idxs = merged_df['agn_flag']==0
    ha_detected = merged_df['ha_detflag_sfr'] == 0
    hb_detected = merged_df['hb_detflag_sfr'] == 0
    merged_idxs = np.logical_and(np.logical_and(ha_detected, hb_detected), agn_idxs)
    sample_df = sample_df[merged_idxs]
    linemeas_df2 = linemeas_df2[merged_idxs]
    return sample_df, linemeas_df2

# sample_df = get_mosdef_compare_sample()
