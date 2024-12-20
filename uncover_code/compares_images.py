from uncover_read_data import read_raw_spec, read_prism_lsf, read_spec_cat, read_supercat
from scipy.interpolate import interp1d
from astropy.io import ascii
from fit_emission_uncover_old import line_list, sig_to_velocity, velocity_to_sig
import numpy as np
from simple_make_dustmap import get_cutout_segmap, find_filters_around_line, get_cutout
from astropy.visualization import make_lupton_rgb
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u


def check_emission_lsf(id_msa):
    emission_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
    ha_sig = emission_df['sigma'].iloc[0]
    pab_sig = emission_df['sigma'].iloc[1]
    gaussian_vels = [sig_to_velocity(line_list[0][1], ha_sig), sig_to_velocity(line_list[1][1], pab_sig)]

    # Read in the lsf
    lsf = read_prism_lsf()
    # interpolate the lsf to match the wavelengths of the data
    lsf['wave_aa'] = lsf['WAVELENGTH'] * 10000
    interp_lsf = interp1d(lsf['wave_aa'], lsf['R'], kind='linear')
    lsf_FWHMs = [line_list[i][1] / interp_lsf(line_list[i][1]) for i in range(len(line_list))]
    # sigma = wavelength / (R * 2.355)
    lsf_sigs = [lsf_FWHMs[i] / 2.355 for i in range(len(line_list))]
    c = 299792 #km/s
    lsf_sigma_v_kms = [c/(interp_lsf(line_list[i][1])*2.355) for i in range(len(line_list))]
    true_vels = [np.sqrt(gaussian_vels[i]**2 - lsf_sigma_v_kms[i]**2)  for i in range(len(line_list))]


def make_3color(id_msa, line_index = 0, plot = False, image_size=(100,100)): 
    spec_df = read_spec_cat()
    supercat_df = read_supercat()
    row = spec_df[spec_df['id_msa']==id_msa]
    # row = supercat_df[supercat_df['id']==62934]
    obj_ra = row['ra'].iloc[0] * u.deg
    obj_dec = row['dec'].iloc[0] * u.deg
    obj_skycoord = SkyCoord(obj_ra, obj_dec)

    line_name = line_list[line_index][0]

    filt_red, filt_green, filt_blue, all_filts = find_filters_around_line(id_msa, line_index)
    filters = [filt_red, filt_green, filt_blue]


    image_red, wht_image_red, photfnu_red = get_cutout(obj_skycoord, filt_red, size=image_size)
    image_green, wht_image_green, photfnu_green = get_cutout(obj_skycoord, filt_green, size=image_size)
    image_blue, wht_image_blue, photfnu_blue = get_cutout(obj_skycoord, filt_blue, size=image_size)
    images = [image_red, image_green, image_blue]
    wht_images = [wht_image_red, wht_image_green, wht_image_blue]
    photfnus = [photfnu_red, photfnu_green, photfnu_blue]

    obj_segmap = get_cutout_segmap(obj_skycoord, size=image_size)

    # Plotting  single image
    def plot_single_3color():
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
        # fig.savefig(save_folder + f'/{id_msa}_{line_name}.pdf')
        plt.show()
        plt.close('all')
    if plot == True:
        plot_single_3color()
    
    return filters, images, wht_images, obj_segmap, photfnus, all_filts


make_3color(6325, plot=True)
# check_emission_lsf(47875)



