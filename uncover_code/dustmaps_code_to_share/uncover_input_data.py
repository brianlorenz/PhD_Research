from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
import os

# Set the path to the local folder on your computer to store everythign related to this project
home_folder = '/Users/brianlorenz/uncover/Linemaps'

# Catalogs go here
path_to_catalogs = home_folder + '/Catalogs'

# Set the path to the local folder with the images
uncover_image_folder = home_folder + '/Catalogs/psf_matched/'
# The images should look something like uncover_v7.2_abell2744clu_f182m_block40_bcgs_sci_f444w-matched

# set to where you want to save the line coverage file
line_coverage_path = home_folder + '/Data/line_coverage'

# set to where you want to save images
save_figures_path = home_folder + '/Figures'

# Emission line info
halpha_wave = 6564.6
halpha_name = 'Halpha'

# Catalog list
# UNCOVER_v5.2.0_LW_SUPER_CATALOG.fits
# UNCOVER_v5.3.0_LW_SUPER_SPScatalog_spsv1.0.fits
# UNCOVER_v5.2.0_SEGMAP.fits
# bcg_surface_brightness.csv


def make_pd_table_from_fits(fits_file_path):
    """Given a path to a .fits file, this will open it, move the data to pandas, then return it

    Parameters:
    fits_file_path (str): path to the data file

    Returns:
    data_df (pd.DataFrame): dataframe of the fits info
    """
    with fits.open(fits_file_path) as hdu:
        data_loc = hdu[1].data
        data_df = Table(data_loc).to_pandas()
        return data_df


def read_supercat(): # For the uncover "SUPER_CATALOG"
    supercat_loc = f'{path_to_catalogs}/UNCOVER_v5.2.0_LW_SUPER_CATALOG.fits'
    supercat_df = make_pd_table_from_fits(supercat_loc)
    return supercat_df

def read_SPS_cat(): # For the stellar population synthesis
    sps_loc = f'{path_to_catalogs}/UNCOVER_v5.3.0_LW_SUPER_SPScatalog_spsv1.0.fits'
    sps_df = make_pd_table_from_fits(sps_loc)
    return sps_df

def read_segmap():
    segmap_loc = f'{path_to_catalogs}/UNCOVER_v5.2.0_SEGMAP.fits'
    with fits.open(segmap_loc) as hdu:
        segmap = hdu[0].data
        segmap_wcs = WCS(hdu[0].header)
    return segmap, segmap_wcs

def read_line_coverage(line_name): # For the stellar population synthesis
    line_coverage_df = pd.read_csv(f'{line_coverage_path}_{line_name}.csv')
    return line_coverage_df

def read_bcg_surface_brightness():
    bcg_surface_brightness_path =  f'{path_to_catalogs}/bcg_surface_brightness.csv'
    bcg_surface_brightness_df = pd.read_csv(bcg_surface_brightness_path)
    return bcg_surface_brightness_df

def check_and_make_dir(file_path):
    """Checks to see if a directory exists - if not, creates the directory

    Parameters:
    file_path (str): Path to a directory that you wish to create

    Returns:
    """
    if not os.path.exists(file_path):
        os.mkdir(file_path)


def setup_directories(home_folder):
    check_and_make_dir(path_to_catalogs)
    check_and_make_dir(uncover_image_folder)
    check_and_make_dir(home_folder + '/Data')
    check_and_make_dir(save_figures_path)
    check_and_make_dir(save_figures_path + '/three_colors')
    check_and_make_dir(save_figures_path + '/sed_images')
    check_and_make_dir(save_figures_path + f'/sed_images/{halpha_name}_sed_images')
    check_and_make_dir(save_figures_path + f'/linemaps')
    check_and_make_dir(save_figures_path + f'/linemaps/{halpha_name}_linemaps/')
    
    


if __name__ == "__main__":
    setup_directories(home_folder)
    pass