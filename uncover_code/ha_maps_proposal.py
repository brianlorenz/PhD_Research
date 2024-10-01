from make_dust_maps import make_3color, compute_cont_pct
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
# from sedpy import observate
from uncover_make_sed import read_sed
from plot_log_linear_rgb import make_log_rgb
import numpy as np
from matplotlib.colors import Normalize, LogNorm
from plot_vals import scale_aspect
from astropy.io import ascii
import sys
from astropy.visualization import make_lupton_rgb
import matplotlib.patches as patches
from astropy.convolution import Gaussian2DKernel, convolve

add_text = 0

sky_rectangle_box = (150, 40, 10, 40)

text_height = 1.02
text_start_left = 0.15
text_start = 0.01
text_sep = 0.25

# method = 'addition'
method = '8x_addition'
# method = 'fractional'
smooth = 1
same_cbar = 1


def make_ha_map(id_msa, factor=8):
    if add_text:
        image_size=(204, 204)
    else:
        image_size=(84, 84)
    ha_filters, ha_images, wht_ha_images, obj_segmap = make_3color(id_msa, line_index=0, plot=False, image_size=image_size)
    ha_filters = ['f_'+filt for filt in ha_filters]
    ha_images = [image.data for image in ha_images]
    wht_ha_images = [image.data for image in wht_ha_images]
    sed_df = read_sed(id_msa)
    def fint_pct(filts):
        red_row = sed_df[sed_df['filter'] == filts[0]]
        green_row = sed_df[sed_df['filter'] == filts[1]]
        blue_row = sed_df[sed_df['filter'] == filts[2]]
        # breakpoint()
        cont_percentile = compute_cont_pct(blue_row.eff_wavelength.iloc[0], green_row.eff_wavelength.iloc[0], red_row.eff_wavelength.iloc[0], blue_row.flux.iloc[0], red_row.flux.iloc[0])
        return cont_percentile, red_row, green_row, blue_row
    
    ha_cont_pct, ha_red_row, ha_green_row, ha_blue_row = fint_pct(ha_filters)
    ha_cont, ha_linemap, ha_image, ha_linemap_snr = get_cont_and_map_ha_maps(ha_images, wht_ha_images, ha_cont_pct)

    fig = plt.figure(figsize=(17, 6))
    gs = GridSpec(1, 4, left=0.05, right=0.99, bottom=0.1, top=0.90, wspace=0.1, hspace=0.3)
    ax_ha_image = fig.add_subplot(gs[0, 0])
    ax_ha_linemap_ms = fig.add_subplot(gs[0, 1])
    ax_ha_linemap_15 = fig.add_subplot(gs[0, 3])
    ax_ha_linemap_30 = fig.add_subplot(gs[0, 2])
    ax_list = [ax_ha_image, ax_ha_linemap_ms, ax_ha_linemap_15, ax_ha_linemap_30]

    # now blur the images with gaussian noise
    ha_images_15 = blur_gaussian_noise(ha_images, ha_filters, ax_list, time=15, factor=factor)
    ha_images_30 = blur_gaussian_noise(ha_images, ha_filters, ax_list, time=30, factor=factor)
    ha_cont_15, ha_linemap_15, _, _ = get_cont_and_map_ha_maps(ha_images_15, wht_ha_images, ha_cont_pct)
    ha_cont_30, ha_linemap_30, _, _ = get_cont_and_map_ha_maps(ha_images_30, wht_ha_images, ha_cont_pct)


    # Norm values
    cont_lower_pct = 10
    cont_upper_pct = 99.99
    cont_scalea = 1e30
    linemap_lower_pct = 10
    linemap_upper_pct = 99.9
    linemap_scalea = 150
    dustmap_lower_pct = 40
    dustmap_upper_pct = 90
    dustmap_scalea = 100
    cmap='inferno'

    def get_norm(image_map, scalea=1, lower_pct=10, upper_pct=99):
        # imagemap_scaled = np.log(scalea*image_map + 1) / np.log(scalea + 1)  
        # imagemap_scaled = np.emath.logn(1000, image_map)  # = [3, 4] 
        imagemap_gt0 = image_map[image_map>0.0001]
        # imagemap_gt0 = image_map[image_map>0.0001]
        
        # norm = LogNorm(vmin=np.percentile(imagemap_gt0,lower_pct), vmax=np.percentile(imagemap_gt0,upper_pct))
        norm = Normalize(vmin=np.percentile(imagemap_gt0,lower_pct), vmax=np.percentile(imagemap_gt0,upper_pct))
        return norm

    ha_linemap_logscaled = make_log_rgb(ha_linemap, ha_linemap, ha_linemap, scalea=linemap_scalea)[:,:,0]
    ha_linemap_15_logscaled = make_log_rgb(ha_linemap_15, ha_linemap_15, ha_linemap_15, scalea=linemap_scalea)[:,:,0]
    ha_linemap_30_logscaled = make_log_rgb(ha_linemap_30, ha_linemap_30, ha_linemap_30, scalea=linemap_scalea)[:,:,0]
    

    # Test smoothing
    # Smooth the dust map
    sigma = 1.0  # Standard deviation for Gaussian kernel
    kernel = Gaussian2DKernel(sigma)
    if smooth:
        smoothed_ha_linemap_15_logscaled = convolve(ha_linemap_15_logscaled, kernel)
        smoothed_ha_linemap_30_logscaled = convolve(ha_linemap_30_logscaled, kernel)
    else:
        smoothed_ha_linemap_15_logscaled = ha_linemap_15_logscaled
        smoothed_ha_linemap_30_logscaled = ha_linemap_30_logscaled
    # Tests if the smoothed versionhas same std as non-smoothed
    # box_x = sky_rectangle_box[0]
    # box_width = sky_rectangle_box[1]
    # box_y = sky_rectangle_box[2]
    # box_height = sky_rectangle_box[3]
    # smooth_std = np.std(smoothed_ha_linemap_30_logscaled[box_x:box_x+box_width, box_y:box_y+box_height])
    # print(smooth_std)
    # print(smooth_std)
    

    clip_factor = 2
    ha_image = ha_image[clip_factor:-clip_factor,clip_factor:-clip_factor]
    ha_linemap_logscaled = ha_linemap_logscaled[clip_factor:-clip_factor,clip_factor:-clip_factor]
    smoothed_ha_linemap_15_logscaled = smoothed_ha_linemap_15_logscaled[clip_factor:-clip_factor,clip_factor:-clip_factor]
    smoothed_ha_linemap_30_logscaled = smoothed_ha_linemap_30_logscaled[clip_factor:-clip_factor,clip_factor:-clip_factor]

    ha_linemap_norm = get_norm(ha_linemap_logscaled, lower_pct=linemap_lower_pct, upper_pct=linemap_upper_pct)
    ha_linemap_15_norm = get_norm(ha_linemap_15_logscaled, lower_pct=linemap_lower_pct, upper_pct=linemap_upper_pct)
    ha_linemap_30_norm = get_norm(ha_linemap_30_logscaled, lower_pct=linemap_lower_pct, upper_pct=linemap_upper_pct)

    if same_cbar:
        ha_linemap_15_norm = ha_linemap_norm
        ha_linemap_30_norm = ha_linemap_norm

    ax_ha_image.imshow(ha_image)
    ax_ha_linemap_ms.imshow(ha_linemap_logscaled, cmap=cmap, norm=ha_linemap_norm)
    ax_ha_linemap_15.imshow(smoothed_ha_linemap_15_logscaled, cmap=cmap, norm=ha_linemap_15_norm)
    ax_ha_linemap_30.imshow(smoothed_ha_linemap_30_logscaled, cmap=cmap, norm=ha_linemap_30_norm)

    
    ax_ha_image.text(text_start, text_height, f'Image', color='black', fontsize=14, transform=ax_ha_image.transAxes)
    ax_ha_linemap_ms.text(text_start, text_height, f'H$\\alpha$ map', color='black', fontsize=14, transform=ax_ha_linemap_ms.transAxes)
    ax_ha_linemap_15.text(text_start, text_height, f'H$\\alpha$ map, 15 min', color='black', fontsize=14, transform=ax_ha_linemap_15.transAxes)
    ax_ha_linemap_30.text(text_start, text_height, f'H$\\alpha$ map, 30 min', color='black', fontsize=14, transform=ax_ha_linemap_30.transAxes)

    

    for ax in ax_list:
        ax.set_xticks([]); ax.set_yticks([])
        # Create a rectangle
        if add_text:
            rect = patches.Rectangle((sky_rectangle_box[0], sky_rectangle_box[2]), sky_rectangle_box[1], sky_rectangle_box[3], linewidth=2, edgecolor='white', facecolor='none')
            ax.add_patch(rect)
        scale_aspect(ax)

    if add_text == 0 and same_cbar == 0 and smooth == 1:
        fig.savefig(f'/Users/brianlorenz/uncover/Figures/simulated_filters/{id_msa}.pdf')
    elif add_text == 0 and same_cbar == 1 and smooth == 1:
        fig.savefig(f'/Users/brianlorenz/uncover/Figures/simulated_filters/{id_msa}_cbar.pdf')
    else:
        fig.savefig(f'/Users/brianlorenz/uncover/Figures/simulated_filters/{id_msa}_ha_map_{method}.pdf')

def blur_gaussian_noise(images, filters, ax_list, time=0, factor=8):
    filter_mag_df = ascii.read('/Users/brianlorenz/uncover/Data/simulated_filters/simulated_filter_limiting_3sigmagnitudes.csv').to_pandas()
    megascience_mag_df = ascii.read('/Users/brianlorenz/uncover/Data/simulated_filters/megascience_mags.csv').to_pandas()

    megascience_matches = ['f_'+megascience_mag_df['Filter'].iloc[i].lower() for i in range(len(megascience_mag_df))]
    megascience_mag_df['Filter_matched'] = megascience_matches

    filter_name_matches = ['f_'+filter_mag_df['Filter'].iloc[i].lower() for i in range(len(filter_mag_df))]
    filter_mag_df['Filter_matched'] = filter_name_matches
    

    blurred_images = []
    for i in range(len(images)):
        image = images[i]
        filt = filters[i]
        plot = [0, 0]
        mag_to_blur_to = filter_mag_df[filter_mag_df['Filter_matched'] == filt].iloc[0][f'mag_{time}min']
        megascience_mag_5sig = megascience_mag_df[megascience_mag_df['Filter_matched'] == filt].iloc[0][f'mag_5sig']
        
        if i==1 and time==15:
            plot = [1,0]
            _ = apply_blur(image, mag_to_blur_to, megascience_mag_5sig, ax_list, plot=plot, factor=factor) # Just for plotting
            plot = [2,1]
        if i==1 and time==30:
            plot = [3,1]
        blurred_image = apply_blur(image, mag_to_blur_to, megascience_mag_5sig, ax_list, plot=plot, factor=factor)
        blurred_images.append(blurred_image)

    return blurred_images


def apply_blur(image, mag_to_blur_to, megascience_mag_5sig, ax_list, plot, factor):
    """
    
    plot (list): [plot_index, value to plot (image, blurred image, or both]
    """
    f_targetmag_ujy = mag_to_jy(mag_to_blur_to) * 1e6
    f_targetmag_ujy_1sig = f_targetmag_ujy / 3

    f_megascience_mag_ujy = mag_to_jy(megascience_mag_5sig) * 1e6
    f_megascience_mag_ujy_1sig = f_megascience_mag_ujy / 5 # divide by 5 for megascience

    noise_increase_1sig = f_targetmag_ujy_1sig - f_megascience_mag_ujy_1sig
    if noise_increase_1sig < 0:
        sys.exit('Unexpected - megascience is less deep than new survey')

    box_x = sky_rectangle_box[0]
    box_width = sky_rectangle_box[1]
    box_y = sky_rectangle_box[2]
    box_height = sky_rectangle_box[3]
    sky_pix_std_image = np.std(image[box_x:box_x+box_width, box_y:box_y+box_height])

    if method == 'addition':
        noise_increase_1sig = noise_increase_1sig
    if method == '8x_addition':
        noise_increase_1sig = factor*noise_increase_1sig
    if method == 'fractional':
        f_increase = f_targetmag_ujy_1sig / f_megascience_mag_ujy_1sig
        noise_increase_1sig = (sky_pix_std_image * f_increase) - sky_pix_std_image
    gaussian_noise = np.random.normal(0, noise_increase_1sig, image.shape)
    blurred_image = image + gaussian_noise
    
    
    sky_pix_std_blurred_image = np.std(blurred_image[box_x:box_x+box_width, box_y:box_y+box_height])

    if plot[0] != 0:
        ax = ax_list[plot[0]]
        height_add = 0
        if plot[1] == 0:
            plot_vars = [sky_pix_std_image, f_megascience_mag_ujy_1sig]
        if plot[1] == 1:
            plot_vars = [sky_pix_std_blurred_image, f_targetmag_ujy_1sig]
        
        if add_text:
            ax.text(text_start+0.97, text_height, f'std: {plot_vars[0]:.4f}', color='green', fontsize=14, transform=ax.transAxes, horizontalalignment='right')
            ax.text(text_start+0.97, text_height+0.08, f'1sig std: {plot_vars[1]:.4f}', color='black', fontsize=14, transform=ax.transAxes, horizontalalignment='right')
    return blurred_image

def mag_to_jy(mag):
    f_jy = 10**(-0.4*(mag-8.9))
    return f_jy

def get_cont_and_map_ha_maps(images, wht_images, pct):
    """Finds continuum as the percentile between the other two filters"""
    cont = np.percentile([images[0], images[2]], pct*100, axis=0)
    err_cont = np.sqrt(((1-pct)*(1/np.sqrt(wht_images[0]))))**2 + (pct*(1/np.sqrt(wht_images[2]))**2)
    linemap = images[1] - cont
    err_linemap = np.sqrt(err_cont**2 + np.sqrt(1/wht_images[1])**2)
    linemap_snr = linemap/err_linemap
    image = make_lupton_rgb(images[0], images[1], images[2], stretch=0.25)
    return cont, linemap, image, linemap_snr

## DO WE WANT THEM SAME COLORBAR OR INDIV SCALED?
# make_ha_map(47875, factor=8)
# make_ha_map(32111, factor=18)
make_ha_map(18471, factor=8)