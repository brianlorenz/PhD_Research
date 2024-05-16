from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits, ascii
from astropy import units as u
from astropy.nddata import Cutout2D
from uncover_read_data import read_supercat, read_raw_spec, read_spec_cat
from uncover_make_sed import get_sed
from fit_emission_uncover import line_list
from sedpy import observate
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.visualization import make_lupton_rgb
from uncover_sed_filters import get_filt_cols
from glob import glob
import sys
from fit_emission_uncover import line_list
import numpy as np
import matplotlib as mpl
from plot_vals import scale_aspect

colors = ['red', 'green', 'blue']

def make_all_dustmap(id_msa_list):
    for id_msa in id_msa_list:
        make_dustmap(id_msa)

def make_all_3color(id_msa_list):
    for id_msa in id_msa_list:
        make_3color(id_msa, line_index=0, plot=True)
        make_3color(id_msa, line_index=1, plot=True)

def make_dustmap(id_msa):
    ha_filters, ha_images = make_3color(id_msa, line_index=0, plot=False)
    pab_filters, pab_images = make_3color(id_msa, line_index=1, plot=False)
    ha_filters = ['f_'+filt for filt in ha_filters]
    pab_filters = ['f_'+filt for filt in pab_filters]
    spec_df = read_raw_spec(id_msa)
    sed_df = get_sed(id_msa)
    zqual_df = read_spec_cat()
    redshift = zqual_df[zqual_df['id_msa']==id_msa]['z_spec'].iloc[0]

    # theoretical scalings (to Hb, from naveen's paper)
    ha_factor = 2.79
    pab_factor = 0.155
    cmap='inferno'

    # fig, axarr = plt.subplots(2, 4, figsize=(16, 8))
    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(2, 6, left=0.05, right=0.99, bottom=0.1, top=0.90, wspace=0.01, hspace=0.3)
    ax_ha_sed = fig.add_subplot(gs[0, 0])
    ax_ha_image = fig.add_subplot(gs[0, 1])
    ax_ha_cont = fig.add_subplot(gs[0, 2])
    ax_ha_linemap = fig.add_subplot(gs[0, 3])
    ax_pab_sed = fig.add_subplot(gs[1, 0])
    ax_pab_image = fig.add_subplot(gs[1, 1])
    ax_pab_cont = fig.add_subplot(gs[1, 2])
    ax_pab_linemap = fig.add_subplot(gs[1, 3])
    ax_dustmap = fig.add_subplot(gs[0:, 4:])
    ax_list = [ax_ha_sed,ax_ha_image,ax_ha_cont,ax_ha_linemap,ax_pab_sed,ax_pab_image,ax_pab_cont,ax_pab_linemap,ax_dustmap]
    ha_cont_pct = plot_sed_around_line(ax_ha_sed, ha_filters, sed_df, spec_df, redshift, 0)
    pab_cont_pct = plot_sed_around_line(ax_pab_sed, pab_filters, sed_df, spec_df, redshift, 1)
    
    def get_cont_and_map(images, pct):
        cont = np.percentile([images[0].data, images[2].data], pct*100, axis=0)
        linemap = images[1].data - cont
        image = make_lupton_rgb(images[0].data, images[1].data, images[2].data, stretch=0.5)
        return cont, linemap, image
    def get_dustmap(halpha_map, pabeta_map):
        ha_map_scaled = halpha_map/ha_factor
        pab_map_scaled = pabeta_map/pab_factor
        dustmap = pab_map_scaled / ha_map_scaled
        return dustmap
         
        
    ha_cont, ha_linemap, ha_image = get_cont_and_map(ha_images, ha_cont_pct)
    pab_cont, pab_linemap, pab_image = get_cont_and_map(pab_images, pab_cont_pct)
    dustmap = get_dustmap(ha_linemap, pab_linemap)

    ax_ha_image.imshow(ha_image)
    ax_pab_image.imshow(pab_image)

   




    ax_ha_cont.imshow(ha_cont, vmin=np.percentile(ha_cont,10), vmax=np.percentile(ha_cont,99), cmap=cmap)
    ax_pab_cont.imshow(pab_cont, vmin=np.percentile(pab_cont,10), vmax=np.percentile(pab_cont,99), cmap=cmap)

    vmin = np.percentile(pab_linemap/pab_factor,10)
    vmax = np.percentile(pab_linemap/pab_factor,99)

    ax_ha_linemap.imshow(ha_linemap/ha_factor, vmin=vmin, vmax=vmax, cmap=cmap)
    ax_pab_linemap.imshow(pab_linemap/pab_factor, vmin=vmin, vmax=vmax, cmap=cmap)

    ax_dustmap.imshow(dustmap, vmin=np.percentile(dustmap,10), vmax=np.percentile(dustmap,99), cmap=cmap)

    text_height = 1.02
    text_start_left = 0.15
    text_start = 0.01
    text_sep = 0.25
    def add_filtertext(ax, filters):
        ax.text(text_start_left, text_height, f'{filters[2][2:]}', color='blue', fontsize=14, transform=ax.transAxes)
        ax.text(text_start_left+text_sep, text_height, f'{filters[1][2:]}', color='green', fontsize=14, transform=ax.transAxes)
        ax.text(text_start_left+2*text_sep, text_height, f'{filters[0][2:]}', color='red', fontsize=14, transform=ax.transAxes)
    add_filtertext(ax_ha_sed, ha_filters)
    add_filtertext(ax_pab_sed, pab_filters)
    
    ax_ha_image.text(text_start, text_height, f'Image', color='black', fontsize=14, transform=ax_ha_image.transAxes)
    ax_ha_cont.text(text_start, text_height, f'H$\\alpha$ continuum', color='black', fontsize=14, transform=ax_ha_cont.transAxes)
    ax_ha_linemap.text(text_start, text_height, f'H$\\alpha$ map', color='black', fontsize=14, transform=ax_ha_linemap.transAxes)
    ax_pab_image.text(text_start, text_height, f'Image', color='black', fontsize=14, transform=ax_pab_image.transAxes)
    ax_pab_cont.text(text_start, text_height, f'Pa$\\beta$ continuum', color='black', fontsize=14, transform=ax_pab_cont.transAxes)
    ax_pab_linemap.text(text_start, text_height, f'Pa$\\beta$ map', color='black', fontsize=14, transform=ax_pab_linemap.transAxes)

    for ax in [ax_ha_image, ax_ha_cont, ax_ha_linemap, ax_pab_image, ax_pab_cont, ax_pab_linemap, ax_dustmap]:
        ax.set_xticks([]); ax.set_yticks([])

    ax_ha_sed.text(0.50, 1.15, f'z = {round(redshift,2)}', color='black', fontsize=18, transform=ax_ha_sed.transAxes)
    ax_ha_sed.text(-0.05, 1.15, f'id = {id_msa}', color='black', fontsize=18, transform=ax_ha_sed.transAxes)
    
    for ax in ax_list:
        scale_aspect(ax)
    save_folder = '/Users/brianlorenz/uncover/Figures/dust_maps'
    fig.savefig(save_folder + f'/{id_msa}_dustmap.pdf')

def plot_sed_around_line(ax, filters, sed_df, spec_df, redshift, line_index):
    line_wave_obs = (line_list[line_index][1] * (1+redshift))/1e4
    # Plot the 3 SED points
    for i in range(len(filters)):
        sed_row = sed_df[sed_df['filter'] == filters[i]]
        ax.errorbar(sed_row['eff_wavelength'], sed_row['flux'], yerr = sed_row['err_flux'], color=colors[i], marker='o')
        
        if i == 0:
            red_wave = sed_row['eff_wavelength'].iloc[0]
            red_flux = sed_row['flux'].iloc[0]
        if i == 1:
            green_wave = sed_row['eff_wavelength'].iloc[0]
        if i == 2:
            blue_wave = sed_row['eff_wavelength'].iloc[0]
            blue_flux = sed_row['flux'].iloc[0]

        # Read and plot each filter curve
        sedpy_name = filters[i].replace('f_', 'jwst_')
        sedpy_filt = observate.load_filters([sedpy_name])[0]
        ax.plot(sedpy_filt.wavelength/1e4, sedpy_filt.transmission/5e5, ls='-', marker='None', color=colors[i], lw=1)
    
    # Compute the percentile to use when combining the continuum
    connect_color = 'purple'
    total_wave_diff = blue_wave - red_wave
    line_wave_diff = green_wave - red_wave
    cont_percentile = line_wave_diff/total_wave_diff
    # breakpoint()
    ax.plot([red_wave, blue_wave], [red_flux, blue_flux], marker='None', ls='--', color=connect_color)
    ax.plot(green_wave, np.percentile([red_flux, blue_flux], cont_percentile*100), marker='o', ls='None', color=connect_color)

    # Plot the spectrum
    ax.plot(spec_df['wave'], spec_df['scaled_flux'], ls='-', marker='None', color='black', lw=1, label='Scaled Spectrum')
    
    # Plot cleanup
    ax.set_xlabel('Wavelength (um)', fontsize=14)
    ax.set_ylabel('Flux (Jy)', fontsize=14)
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(0.8*line_wave_obs, 1.2*line_wave_obs)
    return cont_percentile

def make_3color(id_msa, line_index = 0, plot = False): 
    obj_skycoord = get_coords(id_msa)

    line_name = line_list[line_index][0]

    filt_red, filt_green, filt_blue = find_filters_around_line(id_msa, line_index)
    filters = [filt_red, filt_green, filt_blue]

    image_red = get_cutout(obj_skycoord, filt_red)
    image_green = get_cutout(obj_skycoord, filt_green)
    image_blue = get_cutout(obj_skycoord, filt_blue)
    images = [image_red, image_green, image_blue]



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
        fig.savefig(save_folder + f'/{id_msa}_{line_name}.pdf')
        plt.close('all')
    if plot == True:
        plot_single_3color()
    
    return filters, images

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
    filt_names = get_filt_cols(supercat_df, skip_wide_bands=True)
    filt_names.sort()
    zqual_detected_df = ascii.read('/Users/brianlorenz/uncover/zqual_detected.csv').to_pandas()
    zqual_row = zqual_detected_df[zqual_detected_df['id_msa'] == id_msa]
    detected_filt = zqual_row[f'line{line_number}_filt'].iloc[0]
    detected_index = [i for i in range(len(filt_names)) if filt_names[i] == detected_filt][0]
    if detected_index == 0:
        print(f'For {id_msa}, line {line_number} is detected in {detected_filt}, the bluest filter')
        filt_red = filt_names[detected_index+1].split('_')[1]
        filt_green = filt_names[detected_index].split('_')[1]
        filt_blue = filt_names[detected_index].split('_')[1]
        return filt_red, filt_green, filt_blue
    if detected_index == len(filt_names)-1:
        print(f'For {id_msa}, line {line_number} is detected in {detected_filt}, the reddest filter')
        filt_red = filt_names[detected_index].split('_')[1]
        filt_green = filt_names[detected_index].split('_')[1]
        filt_blue = filt_names[detected_index-1].split('_')[1]
        return filt_red, filt_green, filt_blue
    filt_red = filt_names[detected_index+1].split('_')[1]
    filt_green = filt_names[detected_index].split('_')[1]
    filt_blue = filt_names[detected_index-1].split('_')[1]
    
    return filt_red, filt_green, filt_blue

make_dustmap(25147)


# make_3color(6291)
# make_3color(22755)
# make_3color(42203)
# make_3color(42213)