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

c = 299792458 # m/s

def read_spec_cat():
    spec_cat_loc = '/Users/brianlorenz/uncover/Catalogs/UNCOVER_v0.6_zspec_zqual_catalog.fits'
    spec_data_df = make_pd_table_from_fits(spec_cat_loc)
    return spec_data_df

def read_supercat():
    supercat_loc = '/Users/brianlorenz/uncover/Catalogs/UNCOVER_v5.2.0_LW_SUPER_CATALOG.fits'
    supercat_df = make_pd_table_from_fits(supercat_loc)
    return supercat_df

def read_aper_cat(aper_size='048'):
    aper_cat_loc = f'/Users/brianlorenz/uncover/Catalogs/UNCOVER_v5.3.0_LW_D{aper_size}_CATALOG.fits'
    aper_cat_df = make_pd_table_from_fits(aper_cat_loc)
    return aper_cat_df

def read_slit_loc_cat():
    slit_loc = '/Users/brianlorenz/uncover/Catalogs/uncover-msa-DR4-shutter-locations.fits'
    slit_loc_df = make_pd_table_from_fits(slit_loc)
    return slit_loc_df

def read_SPS_cat():
    sps_loc = '/Users/brianlorenz/uncover/Catalogs/msa_UNCOVER_v3.0.0_LW_SUPER_SPScatalog_spsv1.1.fits'
    sps_df = make_pd_table_from_fits(sps_loc)
    return sps_df

def read_lineflux_cat():
    lines_loc = '/Users/brianlorenz/uncover/Catalogs/uncover-msa-full_depth-default_drz-v0.8a-lines.fits'
    lines_df = make_pd_table_from_fits(lines_loc)
    return lines_df

def read_segmap():
    segmap_loc = '/Users/brianlorenz/uncover/Catalogs/UNCOVER_v5.2.0_SEGMAP.fits'
    with fits.open(segmap_loc) as hdu:
        segmap = hdu[0].data
        segmap_wcs = WCS(hdu[0].header)
    return segmap, segmap_wcs

def read_prism_lsf():
    lsf_loc = '/Users/brianlorenz/uncover/Catalogs/jwst_nirspec_prism_disp.fits'
    lsf_df = make_pd_table_from_fits(lsf_loc)
    return lsf_df

def read_raw_spec(id_msa):
    spec_cat = read_spec_cat()
    redshift = spec_cat[spec_cat['id_msa']==id_msa]['z_spec'].iloc[0]
    raw_spec_loc = f'/Users/brianlorenz/uncover/Catalogs/spectra/uncover-v2_prism-clear_2561_{id_msa}.spec.fits'
    spec_df = make_pd_table_from_fits(raw_spec_loc)
    spec_df['flux'] = spec_df['flux']*1e-6 # Convert uJy to Jy
    spec_df['err'] = spec_df['err']*1e-6

    
    spec_df['wave_aa'] = spec_df['wave']*10000
    spec_df['rest_wave_aa'] = spec_df['wave_aa']/(1+redshift)
    
    c = 299792458 # m/s
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

def read_integrated_spec(id_msa):
    integrated_spec_df = ascii.read(f'/Users/brianlorenz/uncover/Data/integrated_specs/{id_msa}_integrated_spec.csv').to_pandas()
    integrated_spec_df['flux_erg_aa'] = integrated_spec_df['integrated_spec_flux_jy'] * (1e-23*1e10*c / (integrated_spec_df['wave_aa']**2))
    return integrated_spec_df


def make_pd_table_from_fits(file_loc):
    with fits.open(file_loc) as hdu:
        data_loc = hdu[1].data
        data_df = Table(data_loc).to_pandas()
        return data_df

make_pd_table_from_fits('/Users/brianlorenz/uncover/Catalogs/uncover-msa-full_depth-default_drz-v0.8a-lines.fits')

if __name__ == "__main__":
    # read_prism_lsf()
    # read_segmap()
    # fig, ax = plt.subplots(figsize=(20,20)) 
    # ax.imshow(image, vmin=np.percentile(image, 10), vmax=np.percentile(image, 75))
    # plt.show()
    # sps_df = read_SPS_cat()
    # breakpoint()
    # int_spec_df = read_integrated_spec(47875)
    # breakpoint()
    # supercat = read_supercat()
    # aper_cat_df = read_aper_cat()
    # breakpoint()
    pass