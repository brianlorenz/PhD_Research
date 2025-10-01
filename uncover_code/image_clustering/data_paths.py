from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from glob import glob
import sys

# Location to save
pixel_sed_save_loc = '/Users/brianlorenz/uncover/Clustering/pixel_seds/'

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
    