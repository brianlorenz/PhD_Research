from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits, ascii
from astropy import units as u
from astropy.nddata import Cutout2D
from uncover_read_data import read_supercat
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb
from uncover_sed_filters import get_filt_cols
from glob import glob
import sys
from fit_emission_uncover import line_list
import numpy as np

def make_all_3color(id_msa_list):
    for id_msa in id_msa_list:
        make_3color(id_msa, line_index=0)
        make_3color(id_msa, line_index=1)


def make_3color(id_msa, line_index = 0): 
    obj_skycoord = get_coords(id_msa)

    line_name = line_list[line_index][0]

    filt_red, filt_green, filt_blue = find_filters_around_line(id_msa, line_index)

    image_red = get_cutout(obj_skycoord, filt_red)
    image_green = get_cutout(obj_skycoord, filt_green)
    image_blue = get_cutout(obj_skycoord, filt_blue)

    cont_estimate = np.mean(np.array([image_red, image_blue]), axis=0)
    linemap = image_green - cont_estimate

    # Plotting
    save_folder = '/Users/brianlorenz/uncover/Figures/three_colors'
    fig, ax = plt.subplots(figsize = (6,6))
    image = make_lupton_rgb(image_red.data, image_green.data, image_blue.data, stretch=0.5)
    ax.imshow(image)
    text_height = 1.02
    text_start = 0.01
    text_sep = 0.2
    ax.text(text_start, text_height, f'{filt_blue}', color='blue', fontsize=14, transform=ax.transAxes)
    ax.text(text_start+text_sep, text_height, f'{filt_green}', color='green', fontsize=14, transform=ax.transAxes)
    ax.text(text_start+2*text_sep, text_height, f'{filt_red}', color='red', fontsize=14, transform=ax.transAxes)
    ax.text(0.85, text_height, f'{line_name}', color='green', fontsize=14, transform=ax.transAxes)
    fig.savefig(save_folder + f'/{id_msa}_{line_name}.pdf')
    
def get_coords(id_msa):
    supercat_df = read_supercat()
    row = supercat_df[supercat_df['id_msa']==id_msa]
    obj_ra = row['ra'].iloc[0] * u.deg
    obj_dec = row['dec'].iloc[0] * u.deg
    obj_skycoord = SkyCoord(obj_ra, obj_dec)
    return obj_skycoord

def load_image(filt):
    image_folder = '/Users/brianlorenz/uncover/Catalogs/psf_matched/'
    # image_str = f'uncover_v7.2_abell2744clu_{filt}_bcgs_sci_f444w-matched.fits'
    image_str = glob(image_folder + 'uncover_v7.*'+'*_abell2744clu_*'+filt+'*sci_f444w-matched.fits')
    if len(image_str) > 1:
        sys.exit(f'Error: multiple images found for filter {filt}')
    if len(image_str) < 1:
        sys.exit(f'Error: no image found for filter {filt}')
    image_str = image_str[0]
    with fits.open(image_str) as hdu:
        image = hdu[0].data
        wcs = WCS(hdu[0].header)
        photflam = hdu[0].header['PHOTFLAM']
        photplam = hdu[0].header['PHOTPLAM']
    return image, wcs

def get_cutout(obj_skycoord, filt):
    image, wcs = load_image(filt)
    size = (100, 100)
    cutout = Cutout2D(image, obj_skycoord, size, wcs=wcs)
    return cutout

def find_filters_around_line(id_msa, line_number):
    """
    
    Parameters:
    id_msa (int):
    line_number (int): index of the line number in line-list, should be saved in the same way in zqual_df

    """
    supercat_df = read_supercat()
    filt_names = get_filt_cols(supercat_df)
    filt_names.sort()
    zqual_detected_df = ascii.read('/Users/brianlorenz/uncover/zqual_detected.csv').to_pandas()
    zqual_row = zqual_detected_df[zqual_detected_df['id_msa'] == id_msa]
    detected_filt = zqual_row[f'line{line_number}_filt'].iloc[0]
    detected_index = [i for i in range(len(filt_names)) if filt_names[i] == detected_filt][0]
    filt_red = filt_names[detected_index+1].split('_')[1]
    filt_green = filt_names[detected_index].split('_')[1]
    filt_blue = filt_names[detected_index-1].split('_')[1]
    
    return filt_red, filt_green, filt_blue

make_3color(6291)
# make_3color(22755)
# make_3color(42203)
# make_3color(42213)