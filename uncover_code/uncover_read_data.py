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
from astropy.coordinates import SkyCoord
import astropy.units as u



c = 299792458 # m/s

def read_spec_cat():
    spec_cat_loc = '/Users/brianlorenz/uncover/Catalogs/UNCOVER_v0.6_zspec_zqual_catalog.fits'
    spec_data_df = make_pd_table_from_fits(spec_cat_loc)
    return spec_data_df

def read_supercat():
    supercat_loc = '/Users/brianlorenz/uncover/Catalogs/UNCOVER_v5.2.0_LW_SUPER_CATALOG.fits'
    supercat_df = make_pd_table_from_fits(supercat_loc)
    return supercat_df

def read_supercat_newids():
    supercat_loc = '/Users/brianlorenz/uncover/Catalogs/uncover-msa-full_depth-SUPER-v1.1-zspec.a_ungraded.fits'
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

def read_SPS_cat_old():
    sps_loc = '/Users/brianlorenz/uncover/Catalogs/msa_UNCOVER_v3.0.0_LW_SUPER_SPScatalog_spsv1.1.fits'
    sps_df = make_pd_table_from_fits(sps_loc)
    return sps_df

def read_SPS_cat():
    sps_loc = '/Users/brianlorenz/uncover/Catalogs/UNCOVER_v5.3.0_LW_SUPER_zspec_dr4_SPScatalog_spsv1.0.fits'
    sps_df = make_pd_table_from_fits(sps_loc)
    return sps_df

def read_SPS_cat_all():
    sps_all_loc = '/Users/brianlorenz/uncover/Catalogs/UNCOVER_v5.3.0_LW_SUPER_SPScatalog_spsv1.0.fits'
    sps_all_df = make_pd_table_from_fits(sps_all_loc)
    return sps_all_df
    

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

def read_bcg_surface_brightness():
    bcg_surface_brightness_loc = '/Users/brianlorenz/uncover/Data/generated_tables/bcg_surface_brightness.csv'
    bcg_surface_brightness_df = ascii.read(bcg_surface_brightness_loc).to_pandas()
    return bcg_surface_brightness_df


def read_raw_spec(id_msa, read_2d=-1, id_redux = -1):
    """
    read_2d options:
        1 - SPEC1D - 1D Spectrum (this is what we normally read)
        2 - SCI - 2D Spectrum, uJy
        3 - WHT - 2D weights, 1/uJy^2
        4 - PROFILE - 2D Profile
        5 - PROF1D - 1D profile, binary table   
        6 - BACKGROUND - 2D Background
        7 - SLITS - Slit information, binary table
    """
    spec_cat = read_spec_cat()
    redshift = spec_cat[spec_cat['id_msa']==id_msa]['z_spec'].iloc[0]
    # raw_spec_loc = f'/Users/brianlorenz/uncover/Catalogs/spectra_old/uncover-v2_prism-clear_2561_{id_msa}.spec.fits'
    raw_spec_loc = f'/Users/brianlorenz/uncover/Catalogs/DR4_spectra/uncover_DR4_prism-clear_2561_{id_msa}.spec.fits'
    if id_redux > -1:
        print(f'reading with id_redux = {id_redux}')
        raw_spec_loc = f'/Users/brianlorenz/uncover/Catalogs/v1.1_specs/SUPER/uncover_prism-clear_2561_{id_redux}.spec.fits'
    if read_2d > -1:
        with fits.open(raw_spec_loc) as hdu:            
            spec_2d = hdu[read_2d].data
            return spec_2d
    spec_df = make_pd_table_from_fits(raw_spec_loc)
    spec_df['flux'] = spec_df['flux']*1e-6 # Convert uJy to Jy
    spec_df['err'] = spec_df['err']*1e-6

    
    spec_df['wave_aa'] = spec_df['wave']*10000
    spec_df['rest_wave_aa'] = spec_df['wave_aa']/(1+redshift)
    
    c = 299792458 # m/s
    spec_df['flux_erg_aa'] = spec_df['flux'] * (1e-23*1e10*c / (spec_df['wave_aa']**2))
    
    spec_df['rest_flux_erg_aa'] = spec_df['flux_erg_aa'] * (1+redshift)
    spec_df['err_rest_flux_erg_aa'] = spec_df['err'] * (1e-23*1e10*c / (spec_df['wave_aa']**2)) * (1+redshift)


    # if os.path.exists('/Users/brianlorenz/uncover/Figures/spec_sed_compare/compare_ratio.csv'):
    #     spec_df = correct_spec_to_sed(id_msa, spec_df)
    return spec_df

def read_fluxcal_spec(id_msa):
    spec_df_loc = f'/Users/brianlorenz/uncover/Data/fluxcal_specs/{id_msa}_fluxcal_spec.csv'
    spec_df = ascii.read(spec_df_loc).to_pandas()
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

def match_supercat(id_msa):
    spec_df = read_spec_cat()
    supercat = read_supercat()

    target_ra = spec_df[spec_df['id_msa'] == id_msa]['ra'].iloc[0]  
    target_dec = spec_df[spec_df['id_msa'] == id_msa]['dec'].iloc[0]
    target_coord = SkyCoord(ra=target_ra * u.deg, dec=target_dec * u.deg)

    ra_array = supercat['ra'].to_numpy()
    dec_array = supercat['dec'].to_numpy()
    sky_coords = SkyCoord(ra=ra_array * u.deg, dec=dec_array * u.deg)
    separations = target_coord.separation(sky_coords)
    closest_index = np.argmin(separations)
    closest_object = supercat.iloc[closest_index]
    print("Separation:", separations[closest_index])


def make_pd_table_from_fits(file_loc):
    with fits.open(file_loc) as hdu:
        data_loc = hdu[1].data
        data_df = Table(data_loc).to_pandas()
        return data_df
    
def get_id_msa_list(full_sample=False):
    if full_sample:
        id_msa_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/total_before_cuts.csv').to_pandas()
        id_msa_list = id_msa_df['id_msa'].tolist()
        id_msa_skip_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/id_msa_skipped.csv').to_pandas()
        id_msa_skips = id_msa_skip_df['id_msa'].tolist()
        id_msa_list = [x for x in id_msa_list if x not in id_msa_skips]
        id_msa_filter_edge_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/filt_edge.csv').to_pandas()
        id_msa_filt_edge = id_msa_filter_edge_df['id_msa'].tolist()
        id_msa_list = [x for x in id_msa_list if x not in id_msa_filt_edge]
    else:
        id_msa_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/main_sample.csv').to_pandas()
        id_msa_list = id_msa_df['id_msa'].tolist()
        
        # # Adding in the ones where we correct for line not fully  - Now with stricter cuts, these are unusable
        # id_msa_line_notfullcover_df = ascii.read('/Users/brianlorenz/uncover/Data/sample_selection/line_notfullcover_df.csv').to_pandas()
        # id_msa_line_notfullcover_list = id_msa_line_notfullcover_df['id_msa'].tolist()
        # id_msa_list = id_msa_list + id_msa_line_notfullcover_list
    return id_msa_list

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
    # match_supercat(6325)
    # match_supercat(42041)
    # supercat = read_supercat()
    # spec_df = read_spec_cat()
    # aper_cat_df = read_aper_cat()
    # sps_cat = read_SPS2_cat()
    # read_raw_spec(47875, read_2d=True, id_redux=1000000304)
    supercat = read_supercat_newids()
    breakpoint()
    pass