from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sedpy import observate
import os
from astropy.io import ascii
from astropy.wcs import WCS



def read_spec_cat():
    spec_cat_loc = '/Users/brianlorenz/uncover/Catalogs/UNCOVER_v0.6_zspec_zqual_catalog.fits'
    spec_data_df = make_pd_table_from_fits(spec_cat_loc)
    return spec_data_df

def read_supercat():
    supercat_loc = '/Users/brianlorenz/uncover/Catalogs/UNCOVER_v5.2.0_LW_SUPER_CATALOG.fits'
    supercat_df = make_pd_table_from_fits(supercat_loc)
    return supercat_df

def read_SPS_cat():
    sps_loc = '/Users/brianlorenz/uncover/Catalogs/msa_UNCOVER_v3.0.0_LW_SUPER_SPScatalog_spsv1.1.fits'
    sps_df = make_pd_table_from_fits(sps_loc)
    return sps_df

def read_segmap():
    segmap_loc = '/Users/brianlorenz/uncover/Catalogs/UNCOVER_v5.2.0_SEGMAP.fits'
    with fits.open(segmap_loc) as hdu:
        segmap = hdu[0].data
        segmap_wcs = WCS(hdu[0].header)
    return segmap, segmap_wcs

def read_raw_spec(id_msa):
    spec_cat = read_spec_cat()
    redshift = spec_cat[spec_cat['id_msa']==id_msa]['z_spec'].iloc[0]
    raw_spec_loc = f'/Users/brianlorenz/uncover/Catalogs/spectra/uncover-v2_prism-clear_2561_{id_msa}.spec.fits'
    spec_df = make_pd_table_from_fits(raw_spec_loc)
    spec_df['flux'] = spec_df['flux']*1e-6 # Convert uJy to Jy
    spec_df['err'] = spec_df['err']*1e-6

    c = 299792458 # m/s
    spec_df['wave_aa'] = spec_df['wave']*10000
    spec_df['rest_wave_aa'] = spec_df['wave_aa']/(1+redshift)
    
    spec_df['flux_erg_aa'] = spec_df['flux'] * (1e-23*1e10*c / (spec_df['wave_aa']**2))
    
    spec_df['rest_flux_erg_aa'] = spec_df['flux_erg_aa'] * (1+redshift)
    spec_df['err_rest_flux_erg_aa'] = spec_df['err'] * (1e-23*1e10*c / (spec_df['wave_aa']**2)) * (1+redshift)


    if os.path.exists('/Users/brianlorenz/uncover/Figures/spec_sed_compare/compare_ratio.csv'):
        spec_df = correct_spec_to_sed(id_msa, spec_df)
    return spec_df

def correct_spec_to_sed(id_msa, spec_df):
    ratio_df = ascii.read('/Users/brianlorenz/uncover/Figures/spec_sed_compare/compare_ratio.csv').to_pandas()
    scale_factor = ratio_df[ratio_df['id_msa']==id_msa].iloc[0]['full_ratio']
    spec_df['scaled_flux'] = spec_df['flux']*scale_factor
    return spec_df



def make_pd_table_from_fits(file_loc):
    with fits.open(file_loc) as hdu:
        data_loc = hdu[1].data
        data_df = Table(data_loc).to_pandas()
        return data_df

# read_segmap()
# fig, ax = plt.subplots(figsize=(20,20)) 
# ax.imshow(image, vmin=np.percentile(image, 10), vmax=np.percentile(image, 75))
# plt.show()
# sps_df = read_SPS_cat()
# breakpoint()
