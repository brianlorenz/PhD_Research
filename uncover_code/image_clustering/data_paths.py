import numpy as np
import os
import sys
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from glob import glob

# Location to save
clustering_folder = '/Users/brianlorenz/uncover/Clustering/'
pixel_sed_save_loc = f'{clustering_folder}pixel_seds/'
sed_save_loc = f'{clustering_folder}seds/'
image_save_dir = f'{clustering_folder}images/'


# Define paths to files here
SUPER_CATALOG_loc = '/Users/brianlorenz/uncover/Catalogs/UNCOVER_v5.2.0_LW_SUPER_CATALOG.fits'
segmap_loc = '/Users/brianlorenz/uncover/Catalogs/UNCOVER_v5.2.0_SEGMAP.fits'
image_folder = '/Users/brianlorenz/uncover/Catalogs/psf_matched/'
def find_image_path(filt):
    image_path = glob(image_folder + 'uncover_v7.*'+'*_abell2744clu_*'+filt+'*sci_f444w-matched.fits')
    wht_image_path = glob(image_folder + 'uncover_v7.*'+'*_abell2744clu_*'+filt+'*wht_f444w-matched.fits')
    if len(image_path) > 1:
        sys.exit(f'Error: multiple images found for filter {filt}')
    if len(wht_image_path) < 1:
        sys.exit(f'Error: no image found for filter {filt}')
    image_str = image_path[0]
    wht_image_str = wht_image_path[0]
    return image_str, wht_image_str


def read_saved_pixels(id_dr3):
    pixel_data = np.load(pixel_sed_save_loc + f'{id_dr3}_pixels.npz')
    # CONTAINS:
    # pixel_seds = pixel_data['pixel_seds'] # shape of (n_images, pixel_ids)
    # masked_indicies = pixel_data['masked_indicies'] # shape of (2, pixel_ids)
    # image_cutouts = pixel_data['image_cutouts'] # shape of (n_images, cutout_y_size, cutout_x_size)
    # noise_cutouts = pixel_data['noise_cutouts'] # shape of (n_images, cutout_y_size, cutout_x_size)
    # boolean_segmap = pixel_data['boolean_segmap'] # shape of (cutout_y_size, cutout_x_size)
    # obj_segmap = pixel_data['obj_segmap'] # shape of (cutout_y_size, cutout_x_size)
    # filter_names = pixel_data['filter_names'] # shape of (n_images,)
    return pixel_data

def read_sed(id_dr3):
    sed_data = np.load(sed_save_loc + f'{id_dr3}_sed.npz')
    # CONTAINS:
    # sed = pixel_data['sed']
    # err_sed = pixel_data['err_sed'] 
    return sed_data


def read_supercat():
    supercat_df = make_pd_table_from_fits(SUPER_CATALOG_loc)
    return supercat_df

def read_segmap():
    with fits.open(segmap_loc) as hdu:
        segmap = hdu[0].data
        segmap_wcs = WCS(hdu[0].header)
    return segmap, segmap_wcs


def make_pd_table_from_fits(file_loc):
    with fits.open(file_loc) as hdu:
        data_loc = hdu[1].data
        data_df = Table(data_loc).to_pandas()
        return data_df
    

def get_cluster_save_path(cluster_method, norm_method='', id_dr3=-1):
    save_path = pixel_sed_save_loc + f'{cluster_method}{norm_method}/'
    if id_dr3 >= 0:
        save_path = pixel_sed_save_loc + f'{cluster_method}{norm_method}/{id_dr3}_clustered.npz'
    return save_path

def check_and_make_dir(file_path):
    if not os.path.exists(file_path):
        os.mkdir(file_path)