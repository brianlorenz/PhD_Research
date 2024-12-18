from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits, ascii
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.convolution import Gaussian2DKernel, convolve
from uncover_read_data import read_supercat, read_raw_spec, read_spec_cat, read_segmap, read_SPS_cat, read_aper_cat, read_fluxcal_spec
from uncover_make_sed import read_sed
from uncover_sed_filters import unconver_read_filters
from fit_emission_uncover import line_list
from sedpy import observate
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from astropy.visualization import make_lupton_rgb
from uncover_sed_filters import get_filt_cols
from glob import glob
import sys
import pandas as pd
import numpy as np
import matplotlib as mpl
from plot_vals import scale_aspect
from scipy import ndimage
from scipy.signal import convolve2d
from matplotlib.colors import Normalize, LogNorm
from matplotlib.ticker import LogLocator, LogFormatterSciNotation
from compute_av import ha_factor, pab_factor, compute_ratio_from_av, compute_ha_pab_av, compute_ha_pab_av_from_dustmap, read_catalog_av
from plot_log_linear_rgb import make_log_rgb
from dust_equations_prospector import dust2_to_AV
from filter_integrals import integrate_filter, get_transmission_at_line, get_line_coverage
from uncover_prospector_seds import read_prospector
from shutter_loc import plot_shutter_pos
from copy import copy, deepcopy


colors = ['red', 'green', 'blue']
connect_color = 'green'



def make_dustmap_simple(id_msa, aper_size='None', cor_helium=False):
    ha_snr_cut = 1
    pab_snr_cut = 0.5


    # Read in the images
    ha_filters, ha_images, wht_ha_images, obj_segmap, ha_photfnus, ha_all_filts = make_3color(id_msa, line_index=0, plot=False)
    pab_filters, pab_images, wht_pab_images, obj_segmap, pab_photfnus, pab_all_filts = make_3color(id_msa, line_index=1, plot=False)
    ha_sedpy_name = ha_filters[1].replace('f', 'jwst_f')
    ha_sedpy_filt = observate.load_filters([ha_sedpy_name])[0]
    ha_filter_width = ha_sedpy_filt.rectangular_width
    pab_sedpy_name = pab_filters[1].replace('f', 'jwst_f')
    pab_sedpy_filt = observate.load_filters([pab_sedpy_name])[0]
    pab_filter_width = pab_sedpy_filt.rectangular_width

    ha_red_sedpy_name = ha_filters[0].replace('f', 'jwst_f')
    ha_red_sedpy_filt = observate.load_filters([ha_red_sedpy_name])[0]
    pab_red_sedpy_name = pab_filters[0].replace('f', 'jwst_f')
    pab_red_sedpy_filt = observate.load_filters([pab_red_sedpy_name])[0]
    ha_blue_sedpy_name = ha_filters[2].replace('f', 'jwst_f')
    ha_blue_sedpy_filt = observate.load_filters([ha_blue_sedpy_name])[0]
    pab_blue_sedpy_name = pab_filters[2].replace('f', 'jwst_f')
    pab_blue_sedpy_filt = observate.load_filters([pab_blue_sedpy_name])[0]

    ha_rest_wavelength = line_list[0][1]
    pab_rest_wavelength = line_list[1][1]


    # If either halpha or pab is detected in the end filters, decide what to do
    if ha_all_filts == False or pab_all_filts == False:
        print("One of the lines not detected in all filters")
        print("Consider different cont measurement method")
        print("Exiting")
        sys.exit("")

    # Read in filters and redshift
    ha_filters = ['f_'+filt for filt in ha_filters]
    pab_filters = ['f_'+filt for filt in pab_filters]
    spec_df = read_fluxcal_spec(id_msa)
    sed_df = read_sed(id_msa, aper_size=aper_size)
    zqual_df = read_spec_cat()
    redshift = zqual_df[zqual_df['id_msa']==id_msa]['z_spec'].iloc[0]
    
    # Make sure all of the deesignated filters have data
    confirm_filters_not_NaN(id_msa, sed_df, ha_filters, pab_filters)

    # Emission fit properties
    fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
    ha_flux_fit = fit_df.iloc[0]['flux']
    pab_flux_fit = fit_df.iloc[1]['flux']
    ha_sigma = fit_df.iloc[0]['sigma'] # full width of the line
    pab_sigma = fit_df.iloc[1]['sigma'] # full width of the line
    # Helium fit properties, if correcting for it
    if cor_helium:
        helium_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/helium/{id_msa}_emission_fits_helium.csv').to_pandas()
        he_flux = helium_df.iloc[1]['flux']
        pab_cor_helium_factor = pab_flux_fit / (pab_flux_fit + he_flux)

    # Check the coverage fraction of the lines - we want it high in the line, but 0 int he continuum filters
    ha_avg_transmission = get_line_coverage(ha_sedpy_filt, line_list[0][1] * (1+redshift), ha_sigma * (1+redshift))
    pab_avg_transmission = get_line_coverage(pab_sedpy_filt, line_list[1][1] * (1+redshift), pab_sigma * (1+redshift))
    ha_red_avg_transmission = get_line_coverage(ha_red_sedpy_filt, line_list[0][1] * (1+redshift), ha_sigma * (1+redshift))
    pab_red_avg_transmission = get_line_coverage(pab_red_sedpy_filt, line_list[1][1] * (1+redshift), pab_sigma * (1+redshift))
    ha_blue_avg_transmission = get_line_coverage(ha_blue_sedpy_filt, line_list[0][1] * (1+redshift), ha_sigma * (1+redshift))
    pab_blue_avg_transmission = get_line_coverage(pab_blue_sedpy_filt, line_list[1][1] * (1+redshift), pab_sigma * (1+redshift))
    ha_transmissions = [ha_red_avg_transmission, ha_avg_transmission, ha_blue_avg_transmission]
    pab_transmissions = [pab_red_avg_transmission, pab_avg_transmission, pab_blue_avg_transmission]

    if ha_avg_transmission < 0.9 or pab_avg_transmission < 0.9:
        print("One of the lines not covered fully in the filters")
        print("Exiting")
        sys.exit("")

    # Segmap matching
    supercat_df = read_supercat()
    supercat_row = supercat_df[supercat_df['id_msa']==id_msa]
    aperture = supercat_row['use_aper'].iloc[0] # arcsec
    if aper_size != 'None':
        aperture = float(aper_size) / 100
    id_dr3 = supercat_row['id'].iloc[0]
    segmap_idxs = obj_segmap.data == id_dr3
    kernel = np.asarray([[False, True, False],
                     [True, True, True],
                     [False, True, False]])
    # dilated_segmap_idxs = ndimage.binary_dilation(segmap_idxs, kernel)
    eroded_segmap_idxs = ndimage.binary_erosion(segmap_idxs, kernel)
    for i in range(10):
        eroded_segmap_idxs = ndimage.binary_erosion(eroded_segmap_idxs, kernel)
    dilated_segmap_idxs = convolve2d(segmap_idxs.astype(int), kernel.astype(int), mode='same').astype(bool)

    # Read the AV from the catalog
    av_16, av_50, av_84 = read_catalog_av(id_msa, zqual_df)
    av_lineratio = compute_ratio_from_av(av_50)


    cmap='inferno'

    # Set up axes
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
    ax_dustmap = fig.add_subplot(gs[0, 5])
    ax_segmap = fig.add_subplot(gs[1, 5])
    ax_ha_snr = fig.add_subplot(gs[0, 4])
    ax_pab_snr = fig.add_subplot(gs[1, 4])
    ax_list = [ax_ha_sed,ax_ha_image,ax_ha_cont,ax_ha_linemap,ax_pab_sed,ax_pab_image,ax_pab_cont,ax_pab_linemap,ax_dustmap,ax_segmap,ax_ha_snr,ax_pab_snr]
    
    ha_cont_pct, ha_sed_lineflux, ha_trasm_flag, ha_boot_lines, ha_sed_fluxes = plot_sed_around_line(ax_ha_sed, ha_filters, sed_df, spec_df, redshift, 0, ha_transmissions, id_msa)
    pab_cont_pct, pab_sed_lineflux, pab_trasm_flag, pab_boot_lines, pab_sed_fluxes = plot_sed_around_line(ax_pab_sed, pab_filters, sed_df, spec_df, redshift, 1, pab_transmissions, id_msa)

    # CONSIDER Correcting the linefluxes here for NII, helium, transmission effects, etc


    # Compute lineratios
    sed_lineratio = compute_lineratio(ha_sed_lineflux, pab_sed_lineflux)
    boot_sed_lineratios = compute_lineratio(ha_boot_lines, pab_boot_lines)
    sed_lineratio_16 = np.percentile(boot_sed_lineratios, 16)
    sed_lineratio_84 = np.percentile(boot_sed_lineratios, 84)
    err_sed_lineratio_low = sed_lineratio - sed_lineratio_16
    err_sed_lineratio_high = sed_lineratio_84 - sed_lineratio
    sed_lineratio_pcts = [sed_lineratio_16, sed_lineratio_84]
    sed_lineratios = [sed_lineratio, err_sed_lineratio_low, err_sed_lineratio_high]
    # And emfit lineratios
    line_ratio_from_emission = fit_df["ha_pab_ratio"].iloc[0]
    err_line_ratio_from_emission_low = fit_df["err_ha_pab_ratio_low"].iloc[0]
    err_line_ratio_from_emission_high = fit_df["err_ha_pab_ratio_high"].iloc[0]
    emission_lineratios = [line_ratio_from_emission, err_line_ratio_from_emission_low, err_line_ratio_from_emission_high]


    # Make linemaps
    # Need to multiply the image fluxes by 1e-8 to turn them from 10nJy to Jy
    jy_convert_factor = 1e-8
    # Get the data values
    ha_red_image_data = jy_convert_factor*ha_images[0].data
    ha_green_image_data = jy_convert_factor*ha_images[1].data
    ha_blue_image_data = jy_convert_factor*ha_images[2].data
    pab_red_image_data = jy_convert_factor*pab_images[0].data
    pab_green_image_data = jy_convert_factor*pab_images[1].data
    pab_blue_image_data = jy_convert_factor*pab_images[2].data
    # Get the noise values
    ha_red_image_noise = jy_convert_factor*(1/np.sqrt(wht_ha_images[0].data))
    ha_green_image_noise = jy_convert_factor*(1/np.sqrt(wht_ha_images[1].data))
    ha_blue_image_noise = jy_convert_factor*(1/np.sqrt(wht_ha_images[2].data))
    pab_red_image_noise = jy_convert_factor*(1/np.sqrt(wht_pab_images[0].data))
    pab_green_image_noise = jy_convert_factor*(1/np.sqrt(wht_pab_images[1].data))
    pab_blue_image_noise = jy_convert_factor*(1/np.sqrt(wht_pab_images[2].data))
    # Get the bootstrapped images
    bootstrap=10
    ha_red_image_boots = [np.random.normal(loc=ha_red_image_data, scale=ha_red_image_noise) for i in range(bootstrap)]
    ha_green_image_boots = [np.random.normal(loc=ha_green_image_data, scale=ha_green_image_noise) for i in range(bootstrap)]
    ha_blue_image_boots = [np.random.normal(loc=ha_blue_image_data, scale=ha_blue_image_noise) for i in range(bootstrap)]
    pab_red_image_boots = [np.random.normal(loc=pab_red_image_data, scale=pab_red_image_noise) for i in range(bootstrap)]
    pab_green_image_boots = [np.random.normal(loc=pab_green_image_data, scale=pab_green_image_noise) for i in range(bootstrap)]
    pab_blue_image_boots = [np.random.normal(loc=pab_blue_image_data, scale=pab_blue_image_noise) for i in range(bootstrap)]

    ha_linemap, ha_contmap = compute_line(ha_cont_pct, ha_red_image_data, ha_green_image_data, ha_blue_image_data, redshift, 0, ha_filter_width, ha_rest_wavelength, images=True)
    ha_image = make_lupton_rgb(ha_images[0].data, ha_images[1].data, ha_images[2].data, stretch=0.25)
    pab_linemap, pab_contmap = compute_line(pab_cont_pct, pab_red_image_data, pab_green_image_data, pab_blue_image_data, redshift, 0, pab_filter_width, pab_rest_wavelength, images=True)
    pab_image = make_lupton_rgb(pab_images[0].data, pab_images[1].data, pab_images[2].data, stretch=0.25)
    # Bootstrap to compute SNR
    ha_linemap_boots = []
    pab_linemap_boots = []
    for i in range(bootstrap):
        ha_linemap_boot, ha_contmap_boot = compute_line(ha_cont_pct, ha_red_image_boots[i], ha_green_image_boots[i], ha_blue_image_boots[i], redshift, 0, ha_filter_width, ha_rest_wavelength, images=True) 
        pab_linemap_boot, pab_contmap_boot = compute_line(pab_cont_pct, pab_red_image_boots[i], pab_green_image_boots[i], pab_blue_image_boots[i], redshift, 0, pab_filter_width, pab_rest_wavelength, images=True)
        ha_linemap_boots.append(ha_linemap_boot)
        pab_linemap_boots.append(pab_linemap_boot)
    ha_linemap_boot_noise = np.std(ha_linemap_boots, axis=0)
    pab_linemap_boot_noise = np.std(pab_linemap_boots, axis=0)
    ha_linemap_snr = ha_linemap / ha_linemap_boot_noise
    pab_linemap_snr = pab_linemap / pab_linemap_boot_noise
    # Filter the maps by SNR
    ha_snr_thresh, ha_snr_idxs = get_snr_cut(ha_linemap_snr, snr_thresh=ha_snr_cut)
    pab_snr_thresh, pab_snr_idxs = get_snr_cut(pab_linemap_snr, snr_thresh=pab_snr_cut)
    snr_idx = np.logical_and(ha_snr_idxs, pab_snr_idxs)


    # Make dustmap
    dustmap = get_dustmap(ha_linemap, pab_linemap, ha_linemap_snr, pab_linemap_snr)
    avg_ha_map = np.mean(ha_linemap[48:52,48:52])
    avg_pab_map = np.mean(pab_linemap[48:52,48:52])


    ### Modify the maps for visualization
    # Set values where both lines are good to 1, else 0
    # Making the binary masks
    ha_linemap_snr_binary = deepcopy(ha_linemap_snr)
    ha_linemap_snr_binary[ha_snr_idxs] = 1
    ha_linemap_snr_binary[~ha_snr_idxs] = 0
    pab_linemap_snr_binary = deepcopy(pab_linemap_snr)
    pab_linemap_snr_binary[pab_snr_idxs] = 1
    pab_linemap_snr_binary[~pab_snr_idxs] = 0
    both_linemap_snr_binary = deepcopy(pab_linemap_snr)
    both_linemap_snr_binary[snr_idx] = 1
    both_linemap_snr_binary[~snr_idx] = 0    


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

    ha_contmap_logscaled = make_log_rgb(ha_contmap, ha_contmap, ha_contmap, scalea=cont_scalea)[:,:,0]
    pab_contmap_logscaled = make_log_rgb(pab_contmap, pab_contmap, pab_contmap, scalea=cont_scalea)[:,:,0]
    ha_linemap_logscaled = make_log_rgb(ha_linemap, ha_linemap, ha_linemap, scalea=linemap_scalea)[:,:,0]
    pab_linemap_logscaled = make_log_rgb(pab_linemap, pab_linemap, pab_linemap, scalea=linemap_scalea)[:,:,0]  
    dustmap_logscaled = make_log_rgb(dustmap, dustmap, dustmap, scalea=dustmap_scalea)[:,:,0]   
    ha_contmap_norm  = get_norm(ha_contmap_logscaled, lower_pct=cont_lower_pct, upper_pct=cont_upper_pct)
    pab_contmap_norm = get_norm(pab_contmap_logscaled, lower_pct=cont_lower_pct, upper_pct=cont_upper_pct)
    ha_linemap_norm = get_norm(ha_linemap_logscaled, lower_pct=linemap_lower_pct, upper_pct=linemap_upper_pct)
    pab_linemap_norm = get_norm(pab_linemap_logscaled, lower_pct=linemap_lower_pct, upper_pct=linemap_upper_pct)
    dustmap_norm = get_norm(dustmap_logscaled, lower_pct=dustmap_lower_pct, upper_pct=dustmap_upper_pct)
    breakpoint()

    # Display the images
    # ax_segmap.imshow(ha_linemap_snr_old)

    ax_ha_image.imshow(ha_image)
    ax_pab_image.imshow(pab_image)
    ax_segmap.imshow(pab_image)
    

    ax_ha_cont.imshow(ha_contmap_logscaled, cmap=cmap, norm=ha_contmap_norm)
    ax_pab_cont.imshow(pab_contmap_logscaled, cmap=cmap, norm=pab_contmap_norm)

    ax_ha_linemap.imshow(ha_linemap_logscaled, cmap=cmap, norm=ha_linemap_norm)
    ax_pab_linemap.imshow(pab_linemap_logscaled,cmap=cmap, norm=pab_linemap_norm)
    ax_ha_snr.imshow(ha_linemap_logscaled, cmap=cmap, norm=ha_linemap_norm)
    ax_pab_snr.imshow(pab_linemap_logscaled,cmap=cmap, norm=pab_linemap_norm)

    ax_dustmap.imshow(dustmap_logscaled, cmap=cmap, norm=dustmap_norm)


    # Plot the aperture
    aperture_circle = plt.Circle((50, 50), aperture/0.04, edgecolor='green', facecolor='None', lw=3)
    ax_ha_linemap.add_patch(aperture_circle)
    aperture_circle = plt.Circle((50, 50), aperture/0.04, edgecolor='green', facecolor='None', lw=3)
    ax_ha_cont.add_patch(aperture_circle)
    aperture_circle = plt.Circle((50, 50), aperture/0.04, edgecolor='green', facecolor='None', lw=3)
    ax_pab_cont.add_patch(aperture_circle)
    aperture_circle = plt.Circle((50, 50), aperture/0.04, edgecolor='green', facecolor='None', lw=3)
    ax_pab_linemap.add_patch(aperture_circle)

    # Plot the slits
    plot_shutter_pos(ax_ha_cont, id_msa, ha_images[1].wcs)
    plot_shutter_pos(ax_ha_linemap, id_msa, ha_images[1].wcs)
    plot_shutter_pos(ax_pab_cont, id_msa, ha_images[1].wcs)
    plot_shutter_pos(ax_pab_linemap, id_msa, ha_images[1].wcs)


    # Dustmap Contours
    x = np.arange(pab_linemap.shape[1])
    y = np.arange(pab_linemap.shape[0])
    X_pab, Y_pab = np.meshgrid(x, y)
    # Set where pab snr is not at least 2, to zero
    pab_linemap_snr_filt = deepcopy(pab_linemap)
    # pab_linemap_snr_filt[~pab_snr_idxs] = 0
    dustmap_snr_filt = deepcopy(dustmap)
    dustmap_snr_filt[~snr_idx] = 0
    # ax_ha_linemap.contour(X_pab, Y_pab, dustmap_snr_filt, levels=[0.5, 1, 1.5, 2], cmap='Greys')
    ax_segmap.contour(X_pab, Y_pab, dustmap_snr_filt, levels=[2, 4, 6, 8], cmap='Greys')

    # Masked points in gray
    combined_mask_ha = make_combined_mask(ha_linemap_snr_binary, dilated_segmap_idxs)
    combined_mask_pab = make_combined_mask(pab_linemap_snr_binary, dilated_segmap_idxs)
    combined_mask_both = make_combined_mask(both_linemap_snr_binary, dilated_segmap_idxs)
    combined_mask_segmap = make_combined_mask(dilated_segmap_idxs, dilated_segmap_idxs)
    
    from matplotlib import colors
    cmap_gray = colors.ListedColormap(['gray'])
    # ax_dustmap.imshow(combined_mask_both, cmap=cmap_gray)
    ax_ha_snr.imshow(combined_mask_ha, cmap=cmap_gray)
    ax_pab_snr.imshow(combined_mask_pab, cmap=cmap_gray)
    ax_segmap.imshow(combined_mask_segmap, cmap=cmap_gray)

    # Labels and such 
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
    ax_dustmap.text(text_start, text_height, f'Dust map', color='black', fontsize=14, transform=ax_dustmap.transAxes)
    ax_segmap.text(text_start, text_height, f'Segmap', color='black', fontsize=14, transform=ax_segmap.transAxes)
    ax_ha_snr.text(text_start, text_height, f'Ha Linemap SNR>{ha_snr_cut}', color='black', fontsize=14, transform=ax_ha_snr.transAxes)
    ax_pab_snr.text(text_start, text_height, f'PaB Linemap SNR>{pab_snr_cut}', color='black', fontsize=14, transform=ax_pab_snr.transAxes)

    # Set tick invisible
    for ax in [ax_ha_image, ax_ha_cont, ax_ha_linemap, ax_pab_image, ax_pab_cont, ax_pab_linemap, ax_dustmap, ax_segmap, ax_ha_snr, ax_pab_snr]:
        ax.set_xticks([]); ax.set_yticks([])

    ax_ha_sed.text(0.50, 1.15, f'z = {round(redshift,2)}', color='black', fontsize=18, transform=ax_ha_sed.transAxes)
    ax_ha_sed.text(-0.05, 1.15, f'id = {id_msa}', color='black', fontsize=18, transform=ax_ha_sed.transAxes)
    ax_ha_image.text(2.2, 1.10, f'Emission fit : {round(line_ratio_from_emission, 2)}', fontsize=14, transform=ax_ha_image.transAxes)
    ax_ha_image.text(1.3, 1.10, f'sed: {round(sed_lineratio, 2)}', fontsize=14, transform=ax_ha_image.transAxes, color='purple')
    ax_ha_image.text(3.5, 1.10, f'Prospector fit: {round((1/av_lineratio), 2)}', fontsize=14, transform=ax_ha_image.transAxes)
    # ax_segmap.text(-0.25, -0.15, f'Ha sigma: {round((ha_sigma), 2)}', fontsize=14, transform=ax_segmap.transAxes)
    # ax_segmap.text(0.5, -0.15, f'PaB sigma: {round((pab_sigma), 2)}', fontsize=14, transform=ax_segmap.transAxes)

    # Save
    for ax in ax_list:
        scale_aspect(ax)
    save_folder = '/Users/brianlorenz/uncover/Figures/dust_maps'
    aper_add_str = ''
    if aper_size != 'None':
        aper_add_str = f'_aper{aper_size}'
    fig.savefig(save_folder + f'/{id_msa}_dustmap{aper_add_str}.pdf')
    plt.show()

    plt.close('all')

    return sed_lineratios, emission_lineratios


def get_norm(image_map, scalea=1, lower_pct=10, upper_pct=99):
        # imagemap_scaled = np.log(scalea*image_map + 1) / np.log(scalea + 1)  
        # imagemap_scaled = np.emath.logn(1000, image_map)  # = [3, 4] 
        imagemap_gt0 = image_map[image_map>0.0001]
        # imagemap_gt0 = image_map[image_map>0.0001]

        # norm = LogNorm(vmin=np.percentile(imagemap_gt0,lower_pct), vmax=np.percentile(imagemap_gt0,upper_pct))
        norm = Normalize(vmin=np.percentile(imagemap_gt0,lower_pct), vmax=np.percentile(imagemap_gt0,upper_pct))
        return norm

def make_3color(id_msa, line_index = 0, plot = False, image_size=(100,100)): 
    obj_skycoord = get_coords(id_msa)

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
        fig.savefig(save_folder + f'/{id_msa}_{line_name}.pdf')
        plt.close('all')
    if plot == True:
        plot_single_3color()
    
    return filters, images, wht_images, obj_segmap, photfnus, all_filts


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
        photfnu = hdu[0].header['PHOTFNU']
    with fits.open(wht_image_str) as hdu_wht:
        wht_image = hdu_wht[0].data
        wht_wcs = WCS(hdu_wht[0].header)  
    return image, wht_image, wcs, wht_wcs, photfnu

def get_cutout(obj_skycoord, filt, size = (100, 100)):
    image, wht_image, wcs, wht_wcs, photfnu = load_image(filt)

    cutout = Cutout2D(image, obj_skycoord, size, wcs=wcs)
    wht_cutout = Cutout2D(wht_image, obj_skycoord, size, wcs=wht_wcs)
    return cutout, wht_cutout, photfnu

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
    zqual_detected_df = ascii.read('/Users/brianlorenz/uncover/zqual_df_ha_detected.csv').to_pandas()
    zqual_row = zqual_detected_df[zqual_detected_df['id_msa'] == id_msa]
    detected_filt = zqual_row[f'line{line_number}_filt'].iloc[0]
    detected_index = [i for i in range(len(filt_names)) if filt_names[i] == detected_filt][0]
    all_filts = True
    if detected_index == 0:
        print(f'For {id_msa}, line {line_number} is detected in {detected_filt}, the bluest filter')
        filt_red = filt_names[detected_index+1].split('_')[1]
        filt_green = filt_names[detected_index].split('_')[1]
        filt_blue = filt_names[detected_index].split('_')[1]
        all_filts = False
        return filt_red, filt_green, filt_blue, all_filts
    if detected_index == len(filt_names)-1:
        print(f'For {id_msa}, line {line_number} is detected in {detected_filt}, the reddest filter')
        filt_red = filt_names[detected_index].split('_')[1]
        filt_green = filt_names[detected_index].split('_')[1]
        filt_blue = filt_names[detected_index-1].split('_')[1]
        all_filts = False
        return filt_red, filt_green, filt_blue, all_filts
    filt_red = filt_names[detected_index+1].split('_')[1]
    filt_green = filt_names[detected_index].split('_')[1]
    filt_blue = filt_names[detected_index-1].split('_')[1]
    
    return filt_red, filt_green, filt_blue, all_filts

def confirm_filters_not_NaN(id_msa, sed_df, ha_filters, pab_filters):
        for j in range(6):
            if j < 3:
                filt_check = ha_filters[j]
            else:
                filt_check = pab_filters[j-3]
            if np.isnan(sed_df[sed_df['filter'] == filt_check]['flux'].iloc[0]) == True:
                raise AssertionError(f'SED in filter {filt_check} for {id_msa} is NaN, exiting')
            
def get_dustmap(halpha_map, pabeta_map, ha_linemap_snr, pab_linemap_snr): 
        dustmap = pabeta_map / halpha_map
        # Set negative points to nonzero values, we take logs during normalization
        dustmap[dustmap<0.00001] = 0.00001
        dustmap[dustmap>200] = 200


        # Convert the dustmap to an AV value
        av_dustmap = compute_ha_pab_av_from_dustmap(dustmap)
        av_dustmap[av_dustmap<0.00001] = -1

        # Anywhere that halpha was not detected but pabeta was detected, set the dustmap to a high value
        av_dustmap = set_dustmap_av(av_dustmap, halpha_map, ha_linemap_snr, pabeta_map, pab_linemap_snr)

        return av_dustmap

def set_dustmap_av(dustmap, ha_linemap, ha_linemap_snr, pab_linemap, pab_linemap_snr):
    ha_nondetect_idx = ha_linemap<0
    pab_detect_idx = pab_linemap_snr>0.5
    both_idx = np.logical_and(ha_nondetect_idx, pab_detect_idx)
    dustmap[both_idx] = 20
    return dustmap


def plot_sed_around_line(ax, filters, sed_df, spec_df, redshift, line_index, transmissions, id_msa, bootstrap=1000, plt_purple_merged_point=1, show_trasm=1):
    # Controls for various elements on the plot
    plt_verbose_text = show_trasm
    plt_sed_points = 1
    plt_filter_curves = 1
    plt_spectrum = 1

    line_wave_rest = line_list[line_index][1]
    line_wave_obs = (line_wave_rest * (1+redshift))/1e4 # micron
    ax.axvline(line_wave_obs, ls='--', color='green') # observed line, Ha or PaB
    ax.axvline(1.2560034*(1+redshift), ls='--', color='magenta') # He II https://www.mpe.mpg.de/ir/ISO/linelists/Hydrogenic.html
    ax.axvline(1.083646*(1+redshift), ls='--', color='magenta') # He I https://iopscience.iop.org/article/10.3847/1538-3881/ab3a31
    ax.axvline(1.094*(1+redshift), ls='--', color='green') # Pa gamma
    # Can check lines here https://linelist.pa.uky.edu/atomic/query.cgi
    
    # Plot the 3 SED points
    for i in range(len(filters)):
        sed_row = sed_df[sed_df['filter'] == filters[i]]
        if plt_sed_points:
            ax.errorbar(sed_row['eff_wavelength'], sed_row['flux'], yerr = sed_row['err_flux'], color=colors[i], marker='o')
        
        if i == 0:
            red_wave = sed_row['eff_wavelength'].iloc[0]
            red_flux = sed_row['flux'].iloc[0]
            err_red_flux = sed_row['err_flux'].iloc[0]
        if i == 1:
            green_wave = sed_row['eff_wavelength'].iloc[0]
            green_flux = sed_row['flux'].iloc[0]
            err_green_flux = sed_row['err_flux'].iloc[0]
        if i == 2:
            blue_wave = sed_row['eff_wavelength'].iloc[0]
            blue_flux = sed_row['flux'].iloc[0]
            err_blue_flux = sed_row['err_flux'].iloc[0]

        # Read and plot each filter curve
        sedpy_name = filters[i].replace('f_', 'jwst_')
        sedpy_filt = observate.load_filters([sedpy_name])[0]
        if plt_filter_curves:
            ax.plot(sedpy_filt.wavelength/1e4, sedpy_filt.transmission/1e6, ls='-', marker='None', color=colors[i], lw=1)
    
    # Compute the percentile to use when combining the continuum
    connect_color = 'purple'
    
    cont_percentile = compute_cont_pct(blue_wave, green_wave, red_wave, blue_flux, red_flux)
    sedpy_name = filters[1].replace('f_', 'jwst_')
    sedpy_line_filt = observate.load_filters([sedpy_name])[0]
    filter_width = sedpy_line_filt.rectangular_width
    line_flux, cont_value = compute_line(cont_percentile, red_flux, green_flux, blue_flux, redshift, 0, filter_width, line_wave_rest)

    boot_lines = []
    if bootstrap > 0:
        for i in range(bootstrap):
            # Remake fluxes:
            if err_red_flux < 0:
                print('NEGATIVE ERROR for bootstrapping - NEED TO FIX')
                err_red_flux = np.abs(err_red_flux)
            if err_green_flux < 0:
                print('NEGATIVE ERROR for bootstrapping - NEED TO FIX')
                err_green_flux = np.abs(err_green_flux)
            if err_blue_flux < 0:
                print('NEGATIVE ERROR for bootstrapping - NEED TO FIX')
                err_blue_flux = np.abs(err_blue_flux)
            boot_red_flux = np.random.normal(loc=red_flux, scale=err_red_flux, size=1)
            boot_green_flux = np.random.normal(loc=green_flux, scale=err_green_flux, size=1)
            boot_blue_flux = np.random.normal(loc=blue_flux, scale=err_blue_flux, size=1)
            boot_line, boot_cont = compute_line(cont_percentile, boot_red_flux, boot_green_flux, boot_blue_flux, redshift, 0, filter_width, line_wave_rest)            
            boot_lines.append(boot_line)
    boot_lines = np.array(boot_lines)

    if plt_purple_merged_point:
        ax.plot([red_wave, blue_wave], [red_flux, blue_flux], marker='None', ls='--', color=connect_color)
        ax.plot(green_wave, cont_value, marker='o', ls='None', color=connect_color)
        ax.plot([green_wave,green_wave], [green_flux, cont_value], marker='None', ls='-', color='green', lw=2)
        
        fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
        line_flux_fit = fit_df.iloc[line_index]['flux']
        line_flux_fit_jy = flux_erg_to_jy(line_flux_fit, line_list[line_index][1])

        # ax.text(0.98, 0.85, f'SED: {line_flux:.2e}', color='black', transform=ax.transAxes, horizontalalignment='right')
        # ax.text(0.98, 0.79, f'EmFit: {line_flux_fit_jy:.2e}', color='black', transform=ax.transAxes, horizontalalignment='right')
        # ax.text(0.98, 0.73, f'SED/Fit: {(line_flux/line_flux_fit_jy):.2f}', color='black', transform=ax.transAxes, horizontalalignment='right')

    # Plot the spectrum
    if plt_spectrum:
        ax.plot(spec_df['wave'], spec_df['flux_calibrated_jy'], ls='-', marker='None', color='black', lw=1, label='Spectrum')

    # # Plot the prospector spectrum
    # if plt_prospector:
    #     ax.plot(prospector_spec_df['wave_um'], prospector_spec_df['spec_scaled_flux'], ls='-', marker='None', color='orange', lw=1, label='Prospector Spec')
    #     ax.plot(prospector_sed_df['weff_um'], prospector_sed_df['spec_scaled_flux'], ls='None', marker='o', color='magenta', lw=1, label='Prospector Sed', zorder=10000000, alpha=0.5)

    # Add transmission info
    red_transmission = transmissions[0]
    line_transmission = transmissions[1]
    blue_transmission = transmissions[2]
    if plt_verbose_text:
        ax.text(0.02, 0.93, f'Avg trasm', color='black', fontsize=9, transform=ax.transAxes)
        ax.text(0.02, 0.86, f'{round(blue_transmission, 2)}', color='blue', fontsize=9, transform=ax.transAxes)
        ax.text(0.02, 0.79, f'{round(line_transmission, 2)}', color='green', fontsize=9, transform=ax.transAxes)
        ax.text(0.02, 0.72, f'{round(red_transmission, 2)}', color='red', fontsize=9, transform=ax.transAxes)
    
    # Flag objects with either low line transmission or high continuum contamination
    trasm_flag = 0
    if line_transmission < 0.9:
        trasm_flag = 1
    if red_transmission > 0.1:
        trasm_flag = 2
    if blue_transmission > 0.1:
        trasm_flag = 3

    # Plot cleanup
    ax.set_xlabel('Wavelength (um)', fontsize=14)
    ax.set_ylabel('Flux (Jy)', fontsize=14)
    ax.tick_params(labelsize=14)
    if plt_verbose_text:
        ax.legend(fontsize=10)
    ax.set_xlim(0.8*line_wave_obs, 1.2*line_wave_obs)
    ax.set_ylim(0, 1.2*np.max(spec_df['flux_calibrated_jy']))
    sed_fluxes = [red_flux, green_flux, blue_flux]
    return cont_percentile, line_flux, trasm_flag, boot_lines, sed_fluxes


def compute_cont_pct(blue_wave, green_wave, red_wave, blue_flux, red_flux):
    total_wave_diff = blue_wave - red_wave
    line_wave_diff = green_wave - red_wave
    cont_percentile = line_wave_diff/total_wave_diff
    if red_flux>blue_flux:
        cont_percentile = 1-cont_percentile
    return cont_percentile


def compute_line(cont_pct, red_flx, green_flx, blue_flx, redshift, raw_transmission, filter_width, line_rest_wave, images=False):
        """
        Fluxes in Jy
        Line rest wave in angstroms
        """
        if images == True:
            cont_value = np.percentile([red_flx, blue_flx], cont_pct*100, axis=0)
        else:
            cont_value = np.percentile([blue_flx, red_flx], cont_pct*100)

        line_value = green_flx - cont_value # Jy

        # Put in erg/s/cm2/Hz
        line_value = line_value * 1e-23
        
        # # Convert from f_nu to f_lambda
        c = 299792458 # m/s
        observed_wave = line_rest_wave * (1+redshift)
        line_value = line_value * ((c*1e10) / (observed_wave)**2) # erg/s/cm2/angstrom
 
        # # Multiply by filter width to just get F
        # Filter width is observed frame width
        line_value = line_value * filter_width  # erg/s/cm2

        # Scale by raw transmission curve
        # line_value = line_value / raw_transmission

        #Scale by transmission
        # line_value = line_value / scaled_transmission
        return line_value, cont_value

def flux_erg_to_jy(line_flux_erg, line_wave):
    c = 299792458 # m/s
    line_flux_jy = line_flux_erg / (1e-23*1e10*c / ((line_wave)**2))
    return line_flux_jy

def compute_lineratio(ha_flux, pab_flux):
    lineratio = ha_flux / pab_flux
    return lineratio

def get_snr_cut(linemap_snr, snr_thresh=2):
    snr_thresh_line = snr_thresh
    snr_idxs = linemap_snr > snr_thresh_line
    return snr_thresh_line, snr_idxs

def make_combined_mask(snr_binary_map, segmap_idxs):
        combined_mask = np.logical_and(snr_binary_map>0, segmap_idxs)
        total_mask = np.ma.masked_where(combined_mask+1 > 1.5, combined_mask+1)
        return total_mask

if __name__ == "__main__":
    # make_all_dustmap()
    # make_all_dustmap(aper_size='048')
    make_dustmap_simple(39744)
    # make_dustmap(38163)