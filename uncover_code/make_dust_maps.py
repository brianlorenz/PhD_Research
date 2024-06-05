from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits, ascii
from astropy import units as u
from astropy.nddata import Cutout2D
from uncover_read_data import read_supercat, read_raw_spec, read_spec_cat, read_segmap, read_SPS_cat
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
from scipy import ndimage
from scipy.signal import convolve2d
from matplotlib.colors import Normalize, LogNorm
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from compute_av import ha_factor, pab_factor, compute_ha_pab_av, compute_ha_pab_av_from_dustmap
from plot_log_linear_rgb import make_log_rgb
from dust_equations_prospector import dust2_to_AV


colors = ['red', 'green', 'blue']

def make_all_dustmap():
    zqual_df_detected = ascii.read('/Users/brianlorenz/uncover/zqual_detected.csv').to_pandas()
    id_msa_list = zqual_df_detected['id_msa']
    for id_msa in id_msa_list:
        make_dustmap(id_msa)

def make_all_3color(id_msa_list):
    for id_msa in id_msa_list:
        make_3color(id_msa, line_index=0, plot=True)
        make_3color(id_msa, line_index=1, plot=True)

def make_dustmap(id_msa):
    # Read in the images
    ha_filters, ha_images, wht_ha_images, obj_segmap = make_3color(id_msa, line_index=0, plot=False)
    pab_filters, pab_images, wht_pab_images, obj_segmap = make_3color(id_msa, line_index=1, plot=False)
    
    # Compute SNR pixel-by-pixel
    def compute_snr_map(images, wht_images):
        snr_maps = [images[i].data / (1/np.sqrt(wht_images[i].data)) for i in range(len(images))]
        return snr_maps
    ha_snr_maps = compute_snr_map(ha_images, wht_ha_images)
    pab_snr_maps = compute_snr_map(pab_images, wht_pab_images)
    
    # Read in filters and redshift
    ha_filters = ['f_'+filt for filt in ha_filters]
    pab_filters = ['f_'+filt for filt in pab_filters]
    spec_df = read_raw_spec(id_msa)
    sed_df = get_sed(id_msa)
    zqual_df = read_spec_cat()
    redshift = zqual_df[zqual_df['id_msa']==id_msa]['z_spec'].iloc[0]

    # Segmap matching
    supercat_df = read_supercat()
    id_dr3 = supercat_df[supercat_df['id_msa']==id_msa]['id'].iloc[0]
    segmap_idxs = obj_segmap.data == id_dr3
    kernel = np.asarray([[False, True, False],
                     [True, True, True],
                     [False, True, False]])
    # dilated_segmap_idxs = ndimage.binary_dilation(segmap_idxs, kernel)
    dilated_segmap_idxs = convolve2d(segmap_idxs.astype(int), kernel.astype(int), mode='same').astype(bool)

    # Read the AV from the catalog
    sps_df = read_SPS_cat()
    id_dr2 = zqual_df[zqual_df['id_msa']==id_msa]['id_DR2'].iloc[0]
    sps_row = sps_df[sps_df['id']==id_dr2]
    dust_50 = sps_row['dust2_50'].iloc[0]
    av_50 = dust2_to_AV(dust_50)
    print(f'A_V 50 for id_msa {id_msa}: {av_50}')

    
    cmap='inferno'

    # Set up axes
    # fig, axarr = plt.subplots(2, 4, figsize=(16, 8))
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 5, left=0.05, right=0.99, bottom=0.1, top=0.90, wspace=0.01, hspace=0.3)
    ax_ha_sed = fig.add_subplot(gs[0, 0])
    ax_ha_image = fig.add_subplot(gs[0, 1])
    ax_ha_cont = fig.add_subplot(gs[0, 2])
    ax_ha_linemap = fig.add_subplot(gs[0, 3])
    ax_pab_sed = fig.add_subplot(gs[1, 0])
    ax_pab_image = fig.add_subplot(gs[1, 1])
    ax_pab_cont = fig.add_subplot(gs[1, 2])
    ax_pab_linemap = fig.add_subplot(gs[1, 3])
    ax_dustmap = fig.add_subplot(gs[0, 4])
    ax_segmap = fig.add_subplot(gs[1, 4])
    ax_list = [ax_ha_sed,ax_ha_image,ax_ha_cont,ax_ha_linemap,ax_pab_sed,ax_pab_image,ax_pab_cont,ax_pab_linemap,ax_dustmap,ax_segmap]
    
    
    def get_cont_and_map(images, wht_images, pct):
        """Finds continuum as the percentile between the other two filters"""
        cont = np.percentile([images[0].data, images[2].data], pct*100, axis=0)
        err_cont = np.sqrt(((1-pct)*(1/np.sqrt(wht_images[0].data))))**2 + (pct*(1/np.sqrt(wht_images[2].data))**2)
        linemap = images[1].data - cont
        err_linemap = np.sqrt(err_cont**2 + np.sqrt(1/wht_images[1].data)**2)
        linemap_snr = linemap/err_linemap
        image = make_lupton_rgb(images[0].data, images[1].data, images[2].data, stretch=0.25)
        return cont, linemap, image, linemap_snr
    
    def get_dustmap(halpha_map, pabeta_map): # Ha should be 18 times larger than pab, but it's only 3. Leading to huge Avs
        ha_map_scaled = halpha_map/ha_factor
        pab_map_scaled = pabeta_map/pab_factor
        dustmap = pab_map_scaled / ha_map_scaled
        # av_dustmap = compute_ha_pab_av_from_dustmap(dustmap)
        # dustmap = pab_map_scaled - ha_map_scaled
        # breakpoint()
        return dustmap
    
    # Make SED plot, return percentile of line between the other two filters
    ha_cont_pct = plot_sed_around_line(ax_ha_sed, ha_filters, sed_df, spec_df, redshift, 0)
    pab_cont_pct = plot_sed_around_line(ax_pab_sed, pab_filters, sed_df, spec_df, redshift, 1)
    # Make linemaps
    ha_cont, ha_linemap, ha_image, ha_linemap_snr = get_cont_and_map(ha_images, wht_ha_images, ha_cont_pct)
    pab_cont, pab_linemap, pab_image, pab_linemap_snr = get_cont_and_map(pab_images, wht_pab_images, pab_cont_pct)
    
    # Make dustmap
    dustmap = get_dustmap(ha_linemap, pab_linemap)

    # Set negative points to nonzero values, we take logs during normalization. All calculations are complete by now
    # ha_cont[ha_cont<0] = 0.00001
    # pab_cont[pab_cont<0] = 0.00001
    # ha_linemap[ha_linemap<0] = 0.00001
    # pab_linemap[pab_linemap<0] = 0.00001
    # dustmap[dustmap<0.00001] = 0.00001
    
    
    # SHR calculations, need to check these
    # ax_segmap.imshow(segmap_idxs)
    def get_snr_cut(linemap_snr, snr_thresh=80):
        snr_thresh_line = np.percentile(linemap_snr, snr_thresh)
        snr_idxs = linemap_snr > snr_thresh_line
        return snr_thresh_line, snr_idxs
    ha_snr_thresh, ha_snr_idxs = get_snr_cut(ha_linemap_snr)
    pab_snr_thresh, pab_snr_idxs = get_snr_cut(pab_linemap_snr)
    snr_idx = np.logical_or(ha_snr_idxs, pab_snr_idxs)
    snr_idx = ha_snr_maps[1] > np.percentile(ha_snr_maps[1], 70)
    ha_linemap_snr[snr_idx] = 1
    ha_linemap_snr[~snr_idx] = 0
    dustmap[~snr_idx]=0

    

    
    

    def get_norm(image_map, scalea=1, lower_pct=10, upper_pct=99):
        # imagemap_scaled = np.log(scalea*image_map + 1) / np.log(scalea + 1)  
        # imagemap_scaled = np.emath.logn(1000, image_map)  # = [3, 4] 
        imagemap_gt0 = image_map[image_map>0.0001]
        # imagemap_gt0 = image_map[image_map>0.0001]
        
        # norm = LogNorm(vmin=np.percentile(imagemap_gt0,lower_pct), vmax=np.percentile(imagemap_gt0,upper_pct))
        norm = Normalize(vmin=np.percentile(imagemap_gt0,lower_pct), vmax=np.percentile(imagemap_gt0,upper_pct))
        return norm

    
    
    
    # Norm values
    cont_lower_pct = 10
    cont_upper_pct = 99.99
    cont_scalea = 1e30
    linemap_lower_pct = 10
    linemap_upper_pct = 99.9
    linemap_scalea = 150
    dustmap_lower_pct = 10
    dustmap_upper_pct = 97
    dustmap_scalea = 100

    ha_cont_logscaled = make_log_rgb(ha_cont, ha_cont, ha_cont, scalea=cont_scalea)[:,:,0]
    pab_cont_logscaled = make_log_rgb(pab_cont, pab_cont, pab_cont, scalea=cont_scalea)[:,:,0]
    ha_linemap_logscaled = make_log_rgb(ha_linemap, ha_linemap, ha_linemap, scalea=linemap_scalea)[:,:,0]
    pab_linemap_logscaled = make_log_rgb(pab_linemap, pab_linemap, pab_linemap, scalea=linemap_scalea)[:,:,0]  
    dustmap_logscaled = make_log_rgb(dustmap, dustmap, dustmap, scalea=dustmap_scalea)[:,:,0]    
    ha_cont_norm  = get_norm(ha_cont_logscaled, lower_pct=cont_lower_pct, upper_pct=cont_upper_pct)
    pab_cont_norm = get_norm(pab_cont_logscaled, lower_pct=cont_lower_pct, upper_pct=cont_upper_pct)
    ha_linemap_norm = get_norm(ha_linemap_logscaled, lower_pct=linemap_lower_pct, upper_pct=linemap_upper_pct)
    pab_linemap_norm = get_norm(pab_linemap_logscaled, lower_pct=linemap_lower_pct, upper_pct=linemap_upper_pct)
    dustmap_norm = get_norm(dustmap_logscaled, lower_pct=dustmap_lower_pct, upper_pct=dustmap_upper_pct)

    # Colorbar exploreing
    locator = LogLocator(base=2)
    formatter = LogFormatterSciNotation(base=2)
    # cbar = fig.colorbar(ha_cont_show, ax=ax_ha_cont, ticks=locator, format=formatter)    
    # vmin = np.percentile(pab_linemap/pab_factor,lower_pct)
    # vmax = np.percentile(pab_linemap/pab_factor,upper_pct)


    # Display the images
    ax_segmap.imshow(ha_linemap_snr)

    ax_ha_image.imshow(ha_image)
    ax_pab_image.imshow(pab_image)

    ax_ha_cont.imshow(ha_cont_logscaled, cmap=cmap, norm=ha_cont_norm)
    ax_pab_cont.imshow(pab_cont_logscaled, cmap=cmap, norm=pab_cont_norm)

    ax_ha_linemap.imshow(ha_linemap_logscaled, cmap=cmap, norm=ha_linemap_norm)
    ax_pab_linemap.imshow(pab_linemap_logscaled,cmap=cmap, norm=pab_linemap_norm)

    # Showdustmap, masked points in gray
    ax_dustmap.imshow(dustmap_logscaled, cmap=cmap, norm=dustmap_norm)
    masked_dustmap = np.ma.masked_where(ha_linemap_snr+1 > 1.5, ha_linemap_snr+1)
    from matplotlib import colors
    cmap_gray = colors.ListedColormap(['gray'])
    ax_dustmap.imshow(masked_dustmap, cmap=cmap_gray)

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
    
    # Labels
    ax_ha_image.text(text_start, text_height, f'Image', color='black', fontsize=14, transform=ax_ha_image.transAxes)
    ax_ha_cont.text(text_start, text_height, f'H$\\alpha$ continuum', color='black', fontsize=14, transform=ax_ha_cont.transAxes)
    ax_ha_linemap.text(text_start, text_height, f'H$\\alpha$ map', color='black', fontsize=14, transform=ax_ha_linemap.transAxes)
    ax_pab_image.text(text_start, text_height, f'Image', color='black', fontsize=14, transform=ax_pab_image.transAxes)
    ax_pab_cont.text(text_start, text_height, f'Pa$\\beta$ continuum', color='black', fontsize=14, transform=ax_pab_cont.transAxes)
    ax_pab_linemap.text(text_start, text_height, f'Pa$\\beta$ map', color='black', fontsize=14, transform=ax_pab_linemap.transAxes)
    ax_dustmap.text(text_start, text_height, f'Dust map', color='black', fontsize=14, transform=ax_dustmap.transAxes)
    ax_segmap.text(text_start, text_height, f'SNR map', color='black', fontsize=14, transform=ax_segmap.transAxes)

    # Set tick invisile
    for ax in [ax_ha_image, ax_ha_cont, ax_ha_linemap, ax_pab_image, ax_pab_cont, ax_pab_linemap, ax_dustmap, ax_segmap]:
        ax.set_xticks([]); ax.set_yticks([])

    ax_ha_sed.text(0.50, 1.15, f'z = {round(redshift,2)}', color='black', fontsize=18, transform=ax_ha_sed.transAxes)
    ax_ha_sed.text(-0.05, 1.15, f'id = {id_msa}', color='black', fontsize=18, transform=ax_ha_sed.transAxes)
    
    # Save
    for ax in ax_list:
        scale_aspect(ax)
    save_folder = '/Users/brianlorenz/uncover/Figures/dust_maps'
    # plt.show()
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

    

    

    image_red, wht_image_red = get_cutout(obj_skycoord, filt_red)
    image_green, wht_image_green = get_cutout(obj_skycoord, filt_green)
    image_blue, wht_image_blue = get_cutout(obj_skycoord, filt_blue)
    images = [image_red, image_green, image_blue]
    wht_images = [wht_image_red, wht_image_green, wht_image_blue]

    obj_segmap = get_cutout_segmap(obj_skycoord)


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
    
    return filters, images, wht_images, obj_segmap

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
    wht_image_str = glob(image_folder + 'uncover_v7.*'+'*_abell2744clu_*'+filt+'*wht_f444w-matched.fits')
    if len(image_str) > 1:
        sys.exit(f'Error: multiple images found for filter {filt}')
    if len(image_str) < 1:
        sys.exit(f'Error: no image found for filter {filt}')
    image_str = image_str[0]
    wht_image_str = wht_image_str[0]
    with fits.open(image_str) as hdu:
        image = hdu[0].data
        wcs = WCS(hdu[0].header)
        photflam = hdu[0].header['PHOTFLAM']
        photplam = hdu[0].header['PHOTPLAM']
    with fits.open(wht_image_str) as hdu_wht:
        wht_image = hdu_wht[0].data
        wht_wcs = WCS(hdu_wht[0].header)  
    return image, wht_image, wcs, wht_wcs

def get_cutout(obj_skycoord, filt, size = (100, 100)):
    image, wht_image, wcs, wht_wcs = load_image(filt)
    cutout = Cutout2D(image, obj_skycoord, size, wcs=wcs)
    wht_cutout = Cutout2D(wht_image, obj_skycoord, size, wcs=wht_wcs)
    return cutout, wht_cutout

def get_cutout_segmap(obj_skycoord, size = (100, 100)):
    segmap, segmap_wcs = read_segmap()
    segmap_cutout = Cutout2D(segmap, obj_skycoord, size, wcs=segmap_wcs)
    return segmap_cutout

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

make_all_dustmap()
# make_dustmap(47875)
# make_dustmap(25147)


# make_3color(6291)
# make_3color(22755)
# make_3color(42203)
# make_3color(42213)