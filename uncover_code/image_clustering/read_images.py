# Read in a galaxy by ID and transforms the image and mask with pixels as [20, 1] arrays
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy.io import fits
from astropy.wcs import WCS
from data_paths import read_supercat, read_segmap, find_image_path, pixel_sed_save_loc
import numpy as np 
import time


def prepare_images(id_dr3_list):
    # Read in necessary catalogs
    print ('Reading catalogs and images...')
    supercat_df = read_supercat()
    filter_list = get_filt_cols(supercat_df)

    # Read in image files
    image_dict = {}
    segmap, segmap_wcs = read_segmap()
    image_dict['segmap'] = segmap
    image_dict['segmap_wcs'] = segmap_wcs
    for filt in filter_list:
        image, wht_image, wcs, wht_wcs = load_image(filt)
        image_dict[filt] = image
        image_dict[filt+'_wcs'] = wcs
        image_dict[filt+'_wht'] = wht_image
        image_dict[filt+'_wht_wcs'] = wht_wcs


    for id_dr3 in id_dr3_list:
        print(f'Making cutouts for id_dr3 = {id_dr3}...')
        # Read in the image
        cutout_list, wht_cutout_list, boolean_segmap = images_and_segmap(id_dr3, supercat_df, image_dict, filter_list)
        cutout_arr = np.array(cutout_list) # shape is (n_images, cutout_y_size, cutout_x_size)
        wht_cutout_arr = np.array(wht_cutout_list)

        # Mask the image to only get galaxy pixels - currently using segmap
        masked_indicies = np.where(boolean_segmap)

        pixel_seds = cutout_arr[:, masked_indicies[0], masked_indicies[1]] # shape is (n_images, n_pixels) where n_pixels are the number of pixels in the segmap

        # To access a particular pixel's SED, it is: 
        #       pixel_seds[:, pixel_id]
        # To access a pixel's coordinates back on the image, it is
        #       [masked_indicies[0][pixel_id], masked_indicies[1][pixel_id]]
        np.savez(pixel_sed_save_loc + f'{id_dr3}_pixels.npz', pixel_seds=pixel_seds, masked_indicies=masked_indicies, image_cutouts=cutout_arr, wht_image_cutouts=wht_cutout_arr, boolean_segmap=boolean_segmap, filter_names=np.array(filter_list)) 
        breakpoint()

def images_and_segmap(id_dr3, supercat_df, image_dict, filter_list, plot=False): 
    """Read in the segmap and images in each fiilter
    """
    # Get an astropy SkyCoord object cenetered on the galaxy
    obj_skycoord = get_coords(id_dr3, supercat_df)

    # Read in segmap, and use it to determine the size needed for future images
    obj_segmap = get_cutout_segmap(image_dict, obj_skycoord, size=(250,250))
    boolean_segmap = obj_segmap.data==id_dr3
    cutout_size = find_cutout_size(boolean_segmap)
    obj_segmap_sizematch = get_cutout_segmap(image_dict, obj_skycoord, size=cutout_size)
    boolean_segmap_sizematch = obj_segmap_sizematch.data==id_dr3

    cutout_list = []
    wht_cutout_list = []
    for filt in filter_list:
        image_cutout, wht_image_cutout = get_cutout(image_dict, obj_skycoord, filt, size=cutout_size)
        cutout_list.append(image_cutout.data)
        wht_cutout_list.append(wht_image_cutout.data)

    # if plot == True:
    #     fig, ax = plt.subplots(figsize = (6,6))
    #     fig.savefig(figure_save_loc + f'three_colors/{id_dr3}_{line_name}_3color.pdf')
        # plt.close('all')

    return cutout_list, wht_cutout_list, boolean_segmap_sizematch


def load_image(filt):
    image_str, wht_image_str = find_image_path(filt)
    with fits.open(image_str) as hdu:
        image = hdu[0].data
        wcs = WCS(hdu[0].header)
    with fits.open(wht_image_str) as hdu_wht:
        wht_image = hdu_wht[0].data
        wht_wcs = WCS(hdu_wht[0].header)  
    return image, wht_image, wcs, wht_wcs

def get_cutout(image_dict, obj_skycoord, filt, size = (100, 100)):
    image = image_dict[filt]
    wcs = image_dict[filt+'_wcs']
    wht_image = image_dict[filt+'_wht']
    wht_wcs = image_dict[filt+'_wht_wcs']
    cutout = Cutout2D(image, obj_skycoord, size, wcs=wcs)
    wht_cutout = Cutout2D(wht_image, obj_skycoord, size, wcs=wht_wcs)
    return cutout, wht_cutout

def get_cutout_segmap(image_dict, obj_skycoord, size = (100, 100)):
    segmap_cutout = Cutout2D(image_dict['segmap'], obj_skycoord, size, wcs=image_dict['segmap_wcs'])
    return segmap_cutout

def get_coords(id_dr3, supercat_df):
    row = supercat_df[supercat_df['id']==id_dr3]
    obj_ra = row['ra'].iloc[0] * u.deg
    obj_dec = row['dec'].iloc[0] * u.deg
    obj_skycoord = SkyCoord(obj_ra, obj_dec)
    return obj_skycoord

def find_cutout_size(arr):
    """Given the boolean segmap array, find the minimum size needed to capture the whole galaxy in the image
    
    Parameters:
    arr (np.array): True where segmap=obj_id, False elsewhere
    
    Returns:
    cutout_size (float): Suggested size of cutout, contains full galaxy with a 5 pixel buffer on each side """
    # Any True in each row/column
    row_any = arr.any(axis=1)
    col_any = arr.any(axis=0)

    # Find first and last row with any True
    row_indices = np.where(row_any)[0]
    col_indices = np.where(col_any)[0]

    first_row = row_indices[0]
    last_row = row_indices[-1]
    first_col = col_indices[0]
    last_col = col_indices[-1]

    if first_row == 0 or first_col == 0 or last_row == len(arr) or last_col == len(arr):
        print('Need larger segmap size!!!') # The segmap is larger than the image size in this case

    middle_pixel = len(arr)/2
    row_dim_max = np.max([middle_pixel - first_row, last_row - middle_pixel])
    col_dim_max = np.max([middle_pixel - first_col, last_col - middle_pixel])
    single_dim_max = np.max([row_dim_max, col_dim_max])
    min_side = single_dim_max*2 + 10 # 10 pixel buffer
    cutout_size = (min_side, min_side)

    return cutout_size

def get_filt_cols(df, skip_wide_bands=False):
    filt_cols = [col[2:] for col in df.columns if 'f_' in col]
    filt_cols = [col for col in filt_cols if 'alma' not in col]
    if skip_wide_bands ==  True:
        filt_cols = [col for col in filt_cols if 'w' not in col]
    return filt_cols


if __name__ == '__main__':
    prepare_images([44283, 30804])