from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits, ascii
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.convolution import Gaussian2DKernel, convolve
from uncover_read_data import read_supercat, read_raw_spec, read_spec_cat, read_segmap, read_SPS_cat, read_aper_cat, read_fluxcal_spec, get_id_msa_list
from uncover_make_sed import read_sed
from uncover_sed_filters import unconver_read_filters
from fit_emission_uncover_wave_divide import line_list
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
from compute_av import ha_factor, pab_factor, compute_ratio_from_av, compute_ha_pab_av, compute_ha_pab_av_from_dustmap, read_catalog_av, get_nii_correction, get_fe_correction
from plot_log_linear_rgb import make_log_rgb
from dust_equations_prospector import dust2_to_AV
from filter_integrals import integrate_filter, get_transmission_at_line, get_line_coverage
from uncover_prospector_seds import read_prospector
from shutter_loc import plot_shutter_pos, check_point_in_shutter, get_scale_factor
from copy import copy, deepcopy
from uncover_cosmo import find_pix_per_kpc, pixel_scale
from scipy.stats import pearsonr
import random
from simple_compute_lineratio import compute_lineratio
from simple_abs_line_correction import fit_absorption_lines


plt_aperture_paper = True

ha_trasm_thresh = 0.8
pab_trasm_thresh = 0.8

show_aper_and_slit = True # Leave this on

add_vj_color = 0




colors = ['red', 'green', 'blue']
connect_color = 'green'

id_msa_image_size_dict = {
    14573:(40,40),
    18471:(60,60), 
    19179:(80,80),
    19896:(70,70),
    25147:(50,50),
    25774:(40,40),
    32111:(50,50),
    34114:(60,60),
    35436:(40,40),
    36689:(40,40),
    38163:(100,100),
    39744:(50,50),
    39855:(50,50),
    42213:(50,50),
    47875:(50,50),
    50000:(50,50)
}

def make_dustmap_simple(id_msa, aper_size='None', axarr_final=[], ax_labels=False, label_str='', fluxcal_str=''):
    ha_snr_cut = 0.5
    pab_snr_cut = 0.5
    
    if len(axarr_final) > 0:
        image_size = id_msa_image_size_dict[id_msa]
    else:
        image_size = (100, 100)

    # Read in UVJ
    supercat_df = read_supercat()
    zqual_df = read_spec_cat()
    redshift = zqual_df[zqual_df['id_msa']==id_msa]['z_spec'].iloc[0]

    UVJ_filt_names, UVJ_images, wht_UVJ_images, UVJ_photfnus = get_uvj_images(supercat_df, redshift, id_msa, image_size=image_size)

    # Read in the images
    ha_filters, ha_images, wht_ha_images, obj_segmap, ha_photfnus, ha_all_filts = make_3color(id_msa, line_index=0, plot=False, image_size=image_size)
    pab_filters, pab_images, wht_pab_images, obj_segmap, pab_photfnus, pab_all_filts = make_3color(id_msa, line_index=1, plot=False, image_size=image_size)
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
        print(f"Halpha: {ha_all_filts}, PaBeta: {pab_all_filts}")
        print("Consider different cont measurement method")
        # print("Exiting")
        # sys.exit("")

    # Read in filters and redshift
    ha_filters = ['f_'+filt for filt in ha_filters]
    pab_filters = ['f_'+filt for filt in pab_filters]
    spec_df = read_fluxcal_spec(id_msa)
    sed_df = read_sed(id_msa, aper_size=aper_size)
    
    
    # Make sure all of the deesignated filters have data
    confirm_filters_not_NaN(id_msa, sed_df, ha_filters, pab_filters)

    # Emission fit prosperties
    fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting{fluxcal_str}/{id_msa}_emission_fits.csv').to_pandas()
    ha_flux_fit = fit_df.iloc[0]['flux']
    pab_flux_fit = fit_df.iloc[1]['flux']
    ha_sigma = fit_df.iloc[0]['sigma'] # full width of the line
    pab_sigma = fit_df.iloc[1]['sigma'] # full width of the line
    ha_eqw_fit = fit_df.iloc[0]['equivalent_width_aa']
    pab_eqw_fit = fit_df.iloc[1]['equivalent_width_aa']
    


    ha_avg_transmission = get_line_coverage(id_msa, ha_sedpy_filt, redshift, line_name='ha', fluxcal_str=fluxcal_str)
    pab_avg_transmission = get_line_coverage(id_msa, pab_sedpy_filt, redshift, line_name='pab', fluxcal_str=fluxcal_str)
    ha_red_avg_transmission = get_line_coverage(id_msa, ha_red_sedpy_filt, redshift, line_name='ha', fluxcal_str=fluxcal_str)
    pab_red_avg_transmission = get_line_coverage(id_msa, pab_red_sedpy_filt, redshift, line_name='pab', fluxcal_str=fluxcal_str)
    ha_blue_avg_transmission = get_line_coverage(id_msa, ha_blue_sedpy_filt, redshift, line_name='ha', fluxcal_str=fluxcal_str)
    pab_blue_avg_transmission = get_line_coverage(id_msa, pab_blue_sedpy_filt, redshift, line_name='pab', fluxcal_str=fluxcal_str)
    ha_transmissions = [ha_red_avg_transmission, ha_avg_transmission, ha_blue_avg_transmission]
    pab_transmissions = [pab_red_avg_transmission, pab_avg_transmission, pab_blue_avg_transmission]
    print(f"Halpha: {ha_avg_transmission}, PaBeta: {pab_avg_transmission}")

    if ha_avg_transmission < ha_trasm_thresh or pab_avg_transmission < pab_trasm_thresh:
        print("One of the lines not covered fully in the filters")
        print(f"Halpha: {ha_avg_transmission}, PaBeta: {pab_avg_transmission}")
        # print("Exiting")
        # sys.exit("")

    # Segmap matching
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
    gs = GridSpec(2, 6, left=0.05, right=0.95, bottom=0.1, top=0.90, wspace=0.01, hspace=0.3)
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
    
    dustmap_cax = fig.add_axes([0.955, 0.55, 0.015, 0.35])


    ha_cont_pct, _, ha_trasm_flag, ha_boot_lines, ha_sed_fluxes, ha_wave_pct = plot_sed_around_line(ax_ha_sed, ha_filters, sed_df, spec_df, redshift, 0, ha_transmissions, id_msa, fluxcal_str=fluxcal_str)
    pab_cont_pct, _, pab_trasm_flag, pab_boot_lines, pab_sed_fluxes, pab_wave_pct = plot_sed_around_line(ax_pab_sed, pab_filters, sed_df, spec_df, redshift, 1, pab_transmissions, id_msa, fluxcal_str=fluxcal_str)

    # Read in the linefluxes from lineflux_df, don't need to be double-computing here, and already corrected 
    lineflux_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df{fluxcal_str}.csv').to_pandas()
    lineflux_row = lineflux_df[lineflux_df['id_msa'] == id_msa]
    ha_sed_lineflux = lineflux_row['ha_sed_flux'].iloc[0]
    nii_cor_ha_sed_lineflux = lineflux_row['nii_cor_ha_sed_flux'].iloc[0]
    fe_cor_pab_sed_lineflux = lineflux_row['fe_cor_pab_sed_flux'].iloc[0]
    nii_cor_ha_boot_lines = ha_boot_lines * get_nii_correction(id_msa)
    fe_cor_pab_boot_lines = [pab_boot_lines[i] * get_fe_correction(id_msa, boot=True) for i in range(len(pab_boot_lines))] # Inflates errors be fecor scatter
    # fe_cor_pab_boot_lines_old = pab_boot_lines * get_fe_correction(id_msa)
    


    def set_negative_lineflux_to_lowerlim(boot_lines, original_flux):        
        for i in range(len(boot_lines)):
            line_flx = boot_lines[i]
            if line_flx < 0:
                boot_lines[i] = 1e-30
        return boot_lines
    
    ha_boot_lines = set_negative_lineflux_to_lowerlim(ha_boot_lines, ha_sed_lineflux)
    nii_cor_ha_boot_lines = set_negative_lineflux_to_lowerlim(nii_cor_ha_boot_lines, nii_cor_ha_sed_lineflux)
    fe_cor_pab_boot_lines = set_negative_lineflux_to_lowerlim(fe_cor_pab_boot_lines, fe_cor_pab_sed_lineflux)

    nii_cor_ha_sed_lineflux_16 = np.percentile(nii_cor_ha_boot_lines, 16)
    nii_cor_ha_sed_lineflux_84 = np.percentile(nii_cor_ha_boot_lines, 84)
    err_nii_cor_ha_sed_lineflux_low = nii_cor_ha_sed_lineflux - nii_cor_ha_sed_lineflux_16
    err_nii_cor_ha_sed_lineflux_high = nii_cor_ha_sed_lineflux_84 - nii_cor_ha_sed_lineflux

    ha_sed_lineflux_16 = np.percentile(ha_boot_lines, 16)
    ha_sed_lineflux_84 = np.percentile(ha_boot_lines, 84)
    err_ha_sed_lineflux_low = ha_sed_lineflux - ha_sed_lineflux_16
    err_ha_sed_lineflux_high = ha_sed_lineflux_84 - ha_sed_lineflux

    fe_cor_pab_sed_lineflux_16 = np.percentile(fe_cor_pab_boot_lines, 16)
    fe_cor_pab_sed_lineflux_84 = np.percentile(fe_cor_pab_boot_lines, 84)
    err_fe_cor_pab_sed_lineflux_low = fe_cor_pab_sed_lineflux - fe_cor_pab_sed_lineflux_16
    err_fe_cor_pab_sed_lineflux_high = fe_cor_pab_sed_lineflux_84 - fe_cor_pab_sed_lineflux
    

    err_sed_linefluxes = [err_nii_cor_ha_sed_lineflux_low, err_nii_cor_ha_sed_lineflux_high, err_fe_cor_pab_sed_lineflux_low, err_fe_cor_pab_sed_lineflux_high, err_ha_sed_lineflux_low, err_ha_sed_lineflux_high]

    # Compute lineratios
    # Need absorption corrections
    ha_absorp_eqw_fit, pab_absorp_eqw_fit = fit_absorption_lines(id_dr3)
    sed_lineratio = compute_lineratio(nii_cor_ha_sed_lineflux, fe_cor_pab_sed_lineflux, ha_eqw_fit, pab_eqw_fit, ha_absorp_eqw_fit, pab_absorp_eqw_fit)
    # Monte Carlo draw for the absorption line strengths
    boot_sed_lineratios = []
    for i in range(len(nii_cor_ha_boot_lines)):
        boot_sed_lineratio = compute_lineratio(nii_cor_ha_boot_lines[i], fe_cor_pab_boot_lines[i], ha_eqw_fit, pab_eqw_fit, ha_absorp_eqw_fit, pab_absorp_eqw_fit)
        boot_sed_lineratios.append(boot_sed_lineratio)
    boot_sed_lineratios = np.array(boot_sed_lineratios)
    sed_lineratio_16 = np.percentile(boot_sed_lineratios, 16)
    sed_lineratio_84 = np.percentile(boot_sed_lineratios, 84)
    err_sed_lineratio_low = sed_lineratio - sed_lineratio_16
    err_sed_lineratio_high = sed_lineratio_84 - sed_lineratio
    if sed_lineratio_84 > 999:
        sed_lineratio_84 = 99
        err_sed_lineratio_high = 99
    sed_lineratio_pcts = [sed_lineratio_16, sed_lineratio_84]
    sed_lineratios = [sed_lineratio, err_sed_lineratio_low, err_sed_lineratio_high]
    # And emfit lineratios
    line_ratio_from_emission = fit_df["ha_pab_ratio"].iloc[0]
    err_line_ratio_from_emission_low = fit_df["err_ha_pab_ratio_low"].iloc[0]
    err_line_ratio_from_emission_high = fit_df["err_ha_pab_ratio_high"].iloc[0]
    emission_lineratios = [line_ratio_from_emission, err_line_ratio_from_emission_low, err_line_ratio_from_emission_high]
    # Then compute AV measurements
    sed_av = compute_ha_pab_av(1/sed_lineratio)
    boot_sed_avs = compute_ha_pab_av(1/boot_sed_lineratios)
    sed_av_16 = np.percentile(boot_sed_avs, 16)
    sed_av_84 = np.percentile(boot_sed_avs, 84)
    err_sed_av_low = sed_av - sed_av_16
    err_sed_av_high = sed_av_84 - sed_av
    if sed_lineratio_84 > 999:
        sed_av_16 = -99
        err_sed_av_low = -99
    sed_avs = [sed_av, err_sed_av_low, err_sed_av_high]
    #And emission fit av
    av_from_emission = fit_df["ha_pab_av"].iloc[0]
    err_av_from_emission_low = fit_df["err_ha_pab_av_low"].iloc[0]
    err_av_from_emission_high = fit_df["err_ha_pab_av_high"].iloc[0]
    emission_avs = [av_from_emission, err_av_from_emission_low, err_av_from_emission_high]
    

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
    ha_image_noises = [ha_red_image_noise, ha_green_image_noise, ha_blue_image_noise]
    pab_red_image_noise = jy_convert_factor*(1/np.sqrt(wht_pab_images[0].data))
    pab_green_image_noise = jy_convert_factor*(1/np.sqrt(wht_pab_images[1].data))
    pab_blue_image_noise = jy_convert_factor*(1/np.sqrt(wht_pab_images[2].data))
    pab_image_noises = [pab_red_image_noise, pab_green_image_noise, pab_blue_image_noise]
    # Get the bootstrapped images
    bootstrap=1000
    ha_red_image_boots = [np.random.normal(loc=ha_red_image_data, scale=ha_red_image_noise) for i in range(bootstrap)]
    ha_green_image_boots = [np.random.normal(loc=ha_green_image_data, scale=ha_green_image_noise) for i in range(bootstrap)]
    ha_blue_image_boots = [np.random.normal(loc=ha_blue_image_data, scale=ha_blue_image_noise) for i in range(bootstrap)]
    pab_red_image_boots = [np.random.normal(loc=pab_red_image_data, scale=pab_red_image_noise) for i in range(bootstrap)]
    pab_green_image_boots = [np.random.normal(loc=pab_green_image_data, scale=pab_green_image_noise) for i in range(bootstrap)]
    pab_blue_image_boots = [np.random.normal(loc=pab_blue_image_data, scale=pab_blue_image_noise) for i in range(bootstrap)]

    ha_linemap, ha_contmap, err_ha_linemap = compute_line(ha_cont_pct, ha_red_image_data, ha_green_image_data, ha_blue_image_data, redshift, 0, ha_filter_width, ha_rest_wavelength, images=True, image_noises=ha_image_noises, wave_pct=ha_wave_pct)
    ha_image = make_lupton_rgb(ha_images[0].data, ha_images[1].data, ha_images[2].data, stretch=0.25)
    pab_linemap, pab_contmap, err_pab_linemap = compute_line(pab_cont_pct, pab_red_image_data, pab_green_image_data, pab_blue_image_data, redshift, 0, pab_filter_width, pab_rest_wavelength, images=True, image_noises=pab_image_noises, wave_pct=pab_wave_pct)
    pab_image = make_lupton_rgb(pab_images[0].data, pab_images[1].data, pab_images[2].data, stretch=0.25)
    # Bootstrap to compute SNR
    ha_linemap_boots = []
    pab_linemap_boots = []
    ha_contmap_boots = []
    for i in range(bootstrap):
        ha_linemap_boot, ha_contmap_boot, _ = compute_line(ha_cont_pct, ha_red_image_boots[i], ha_green_image_boots[i], ha_blue_image_boots[i], redshift, 0, ha_filter_width, ha_rest_wavelength, images=True, image_noises=ha_image_noises, wave_pct=ha_wave_pct) 
        pab_linemap_boot, pab_contmap_boot, _ = compute_line(pab_cont_pct, pab_red_image_boots[i], pab_green_image_boots[i], pab_blue_image_boots[i], redshift, 0, pab_filter_width, pab_rest_wavelength, images=True, image_noises=pab_image_noises, wave_pct=pab_wave_pct)
        ha_linemap_boots.append(ha_linemap_boot)
        pab_linemap_boots.append(pab_linemap_boot)
        ha_contmap_boots.append(ha_contmap_boot)
    ha_linemap_boots_from_err = [np.random.normal(loc=ha_linemap, scale=err_ha_linemap) for i in range(bootstrap)]
    ha_linemap_boot_noise = np.std(ha_linemap_boots, axis=0)
    pab_linemap_boot_noise = np.std(pab_linemap_boots, axis=0)
    ha_linemap_snr = ha_linemap / err_ha_linemap
    pab_linemap_snr = pab_linemap / err_pab_linemap
    # Filter the maps by SNR
    ha_snr_thresh, ha_snr_idxs = get_snr_cut(ha_linemap_snr, snr_thresh=ha_snr_cut)
    pab_snr_thresh, pab_snr_idxs = get_snr_cut(pab_linemap_snr, snr_thresh=pab_snr_cut)
    snr_idx = np.logical_and(ha_snr_idxs, pab_snr_idxs)

    # Make dustmap
    dustmap = get_dustmap(ha_linemap, pab_linemap, ha_linemap_snr, pab_linemap_snr)
    avg_ha_map = np.mean(ha_linemap[48:52,48:52])
    avg_pab_map = np.mean(pab_linemap[48:52,48:52])

    # Make UVJ colormaps
    vj_map = -2.5*np.log10(UVJ_images[1].data/UVJ_images[2].data)
    vj_map = np.nan_to_num(vj_map, nan=-99)


    # Measure the offsets with Wren's method
    snr_thresh_map = 0
    obj_skycoord = get_coords(id_msa)
    image_150w, wht_image_150w, photfnu_150w = get_cutout(obj_skycoord, 'f150w', size=image_size)

    ha_contmap_highsnr_idx = find_pixels_above_sky_noise(ha_contmap, segmap_idxs, snr_thresh_map=snr_thresh_map)
    ha_linemap_highsnr_idx = find_pixels_above_sky_noise(ha_linemap, segmap_idxs, snr_thresh_map=snr_thresh_map)
    f150w_highsnr_idx = find_pixels_above_sky_noise(image_150w.data, segmap_idxs, snr_thresh_map=snr_thresh_map)
    pab_contmap_highsnr_idx = find_pixels_above_sky_noise(pab_contmap, segmap_idxs, snr_thresh_map=snr_thresh_map)
    pab_linemap_highsnr_idx = find_pixels_above_sky_noise(pab_linemap, segmap_idxs, snr_thresh_map=snr_thresh_map)
    pab_linemap_segmap_snrcut_idx = np.logical_and(pab_snr_idxs,pab_linemap_highsnr_idx)

    r_value_info_haline_hacont = plot_and_correlate_highsnr_pix(id_dr3, ha_contmap, ha_contmap_highsnr_idx, 'Ha_cont', ha_linemap, ha_linemap_highsnr_idx, 'Ha_line', snr_thresh_map, bootstrap=bootstrap)
    r_value_info_haline_f150w = plot_and_correlate_highsnr_pix(id_dr3, image_150w.data, f150w_highsnr_idx, 'F150W', ha_linemap, ha_linemap_highsnr_idx, 'Ha_line', snr_thresh_map)
    r_value_info_haline_pabline = plot_and_correlate_highsnr_pix(id_dr3, pab_linemap, pab_linemap_highsnr_idx, 'PaB_line', ha_linemap, ha_linemap_highsnr_idx, 'Ha_line', snr_thresh_map)
    r_value_info_haline_pabline_snrcut = plot_and_correlate_highsnr_pix(id_dr3, pab_linemap, pab_linemap_segmap_snrcut_idx, 'PaB_line', ha_linemap, ha_linemap_highsnr_idx, 'Ha_line', snr_thresh_map)

    r_value_info_pabcont_pabline = plot_and_correlate_highsnr_pix(id_dr3, pab_linemap, pab_linemap_highsnr_idx, 'PaB_line', pab_contmap, pab_contmap_highsnr_idx, 'PaB_continuum', snr_thresh_map)
    r_value_info_pabcont_pabline_snrcut = plot_and_correlate_highsnr_pix(id_dr3, pab_linemap, pab_linemap_segmap_snrcut_idx, 'PaB_line', pab_contmap, pab_contmap_highsnr_idx, 'PaB_continuum', snr_thresh_map)

    r_value_info = r_value_info_haline_hacont
    if bootstrap > 0:
        boot_rs = r_value_info[-1]
        r_value_info = r_value_info[:-1]
        r_val_16_boot = np.percentile(boot_rs, 16)
        r_val_84_boot = np.percentile(boot_rs, 84)
        r_value_info.append(r_val_16_boot)
        r_value_info.append(r_val_84_boot)

    r_values = []
    for i in range(bootstrap):
        id_dr3, r_value, _, _, _ = plot_and_correlate_highsnr_pix(id_dr3, ha_contmap, ha_contmap_highsnr_idx, 'Ha_cont', ha_linemap_boots_from_err[i], ha_linemap_highsnr_idx, 'Ha_line', snr_thresh_map, plot=False)
        r_values.append(r_value)
    r_val_16_mc = np.percentile(r_values, 16)
    r_val_84_mc = np.percentile(r_values, 84)
    r_value_info.append(r_val_16_mc)
    r_value_info.append(r_val_84_mc)

    



    # Now try Mariska's method
    ha_linemap_scaled_to_cont, map_scale_factor, cc_similarity = scale_linemap_to_other(ha_linemap, ha_contmap, segmap_idxs)
    sim_index = compute_similarity_linemaps(ha_linemap_scaled_to_cont, ha_contmap, segmap_idxs)
    sim_index_values = [cc_similarity, sim_index]

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
    # dustmap_logscaled = make_log_rgb(dustmap, dustmap, dustmap, scalea=dustmap_scalea)[:,:,0]   
    ha_contmap_norm  = get_norm(ha_contmap_logscaled, lower_pct=cont_lower_pct, upper_pct=cont_upper_pct)
    pab_contmap_norm = get_norm(pab_contmap_logscaled, lower_pct=cont_lower_pct, upper_pct=cont_upper_pct)
    ha_linemap_norm = get_norm(ha_linemap_logscaled, lower_pct=linemap_lower_pct, upper_pct=linemap_upper_pct)
    pab_linemap_norm = get_norm(pab_linemap_logscaled, lower_pct=linemap_lower_pct, upper_pct=linemap_upper_pct)
    # dustmap_norm = get_norm(dustmap_logscaled, lower_pct=dustmap_lower_pct, upper_pct=dustmap_upper_pct)
    dustmap_norm = Normalize(vmin=0.1, vmax=5)

    # Display the images
    # ax_segmap.imshow(ha_linemap_snr_old)

    ax_ha_image.imshow(ha_image, origin='lower')
    ax_pab_image.imshow(pab_image, origin='lower')
    ax_segmap.imshow(pab_image, origin='lower')
    

    ax_ha_cont.imshow(ha_contmap_logscaled, cmap=cmap, norm=ha_contmap_norm, origin='lower')
    ax_pab_cont.imshow(pab_contmap_logscaled, cmap=cmap, norm=pab_contmap_norm, origin='lower')

    ax_ha_linemap.imshow(ha_linemap_logscaled, cmap=cmap, norm=ha_linemap_norm, origin='lower')
    ax_pab_linemap.imshow(pab_linemap_logscaled,cmap=cmap, norm=pab_linemap_norm, origin='lower')
    ax_ha_snr.imshow(ha_linemap_logscaled, cmap=cmap, norm=ha_linemap_norm, origin='lower')
    ax_pab_snr.imshow(pab_linemap_logscaled,cmap=cmap, norm=pab_linemap_norm, origin='lower')

    dustmap_imshow = ax_dustmap.imshow(dustmap, cmap=cmap, norm=dustmap_norm, origin='lower')
    dustmap_cbar = fig.colorbar(dustmap_imshow, cax=dustmap_cax)
    dustmap_cbar.set_label('Dustmap AV', fontsize=14)
    dustmap_cbar.ax.tick_params(labelsize=14)

    if show_aper_and_slit == True:
        # Plot the aperture
        aperture_circle = plt.Circle((50, 50), aperture/pixel_scale, edgecolor='green', facecolor='None', lw=3)
        # ax_ha_linemap.add_patch(aperture_circle)
        aperture_circle = plt.Circle((50, 50), aperture/pixel_scale, edgecolor='green', facecolor='None', lw=3)
        ax_ha_cont.add_patch(aperture_circle)
        aperture_circle = plt.Circle((50, 50), aperture/pixel_scale, edgecolor='green', facecolor='None', lw=3)
        ax_pab_cont.add_patch(aperture_circle)
        aperture_circle = plt.Circle((50, 50), aperture/pixel_scale, edgecolor='green', facecolor='None', lw=3)
        # ax_pab_linemap.add_patch(aperture_circle)

        # Plot the slits
        vertices_list, cropped_vertices_list, gauss_info = plot_shutter_pos(ax_ha_cont, id_msa, ha_images[1].wcs)
        gauss_x_pixels  = gauss_info[0]
        gauss_y_vals  = gauss_info[1]
        central_vertex_1  = gauss_info[2]
        central_vertex_2  = gauss_info[3]
        # plot_shutter_pos(ax_ha_linemap, id_msa, ha_images[1].wcs)
        plot_shutter_pos(ax_pab_cont, id_msa, ha_images[1].wcs)
        # plot_shutter_pos(ax_pab_linemap, id_msa, ha_images[1].wcs)

    # Dustmap Contours
    x = np.arange(pab_linemap.shape[1])
    y = np.arange(pab_linemap.shape[0])
    X_pab, Y_pab = np.meshgrid(x, y)
    # Set where pab snr is not at least 2, to zero
    pab_linemap_snr_filt = deepcopy(pab_linemap)
    pab_linemap_snr_filt[~pab_snr_idxs] = 0
    dustmap_snr_filt = deepcopy(dustmap)
    dustmap_snr_filt[~snr_idx] = 0

    # Shutter calcs - need to leave the shutter on
    point_in_shutter_arr = np.full(image_size, False, dtype=bool)
    shutter_scale_vals = np.full(image_size, 0, dtype=float)
    for x in range(image_size[0]):
        for y in range(image_size[1]):
            point_in_shutter = False
            for vertices in vertices_list:
                if check_point_in_shutter(x, y, vertices):
                    point_in_shutter = True
            point = np.array([x,y])
            shutter_scale_vals[y][x] = get_scale_factor(point, gauss_x_pixels, gauss_y_vals, central_vertex_1, central_vertex_2)
            point_in_shutter_arr[y][x] = point_in_shutter
            combined_shutter_arr = point_in_shutter_arr*shutter_scale_vals
    # point_in_shutter_arr = np.flip(point_in_shutter_arr, axis=0)
    # point_in_shutter_arr = np.flip(point_in_shutter_arr, axis=1)
    ha_in_shutter = np.sum(ha_linemap * combined_shutter_arr)
    pab_in_shutter = np.sum(pab_linemap * combined_shutter_arr)
    lineratio_in_shutter = pab_in_shutter/ha_in_shutter
    av_in_shutter = compute_ha_pab_av(pab_in_shutter/ha_in_shutter)
    shutter_calcs = [ha_in_shutter, pab_in_shutter, lineratio_in_shutter, av_in_shutter]
    
    

    # Masked points in gray
    combined_mask_ha = make_combined_mask(ha_linemap_snr_binary, dilated_segmap_idxs)
    combined_mask_pab = make_combined_mask(pab_linemap_snr_binary, dilated_segmap_idxs)
    combined_mask_both = make_combined_mask(both_linemap_snr_binary, dilated_segmap_idxs)
    combined_mask_segmap = make_combined_mask(dilated_segmap_idxs, dilated_segmap_idxs)

    # Make a copy of the snr map to be used for contouring
    pab_contour_map = deepcopy(pab_linemap_snr)
    pab_contour_map[~combined_mask_segmap.mask] = 0
    # ax_ha_linemap.contour(X_pab, Y_pab, pab_contour_map, levels=[1,2,3,4,5], cmap='Greys')
    # ax_segmap.contour(X_pab, Y_pab, dustmap_snr_filt, levels=[2, 4, 6, 8], cmap='Greys')
    
    from matplotlib import colors
    cmap_gray = colors.ListedColormap(['gray'])
    ax_dustmap.imshow(combined_mask_both, cmap=cmap_gray, origin='lower')
    ax_ha_snr.imshow(combined_mask_ha, cmap=cmap_gray, origin='lower')
    ax_pab_snr.imshow(combined_mask_pab, cmap=cmap_gray, origin='lower')
    ax_segmap.imshow(combined_mask_segmap, cmap=cmap_gray, origin='lower')

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
    save_folder = f'/Users/brianlorenz/uncover/Figures/dust_maps{fluxcal_str}'
    aper_add_str = ''
    if aper_size != 'None':
        aper_add_str = f'_aper{aper_size}'
    fig.savefig(save_folder + f'/{id_msa}_dustmap{aper_add_str}.pdf')
    # plt.show()


    if len(axarr_final) > 0:
        paper_font = 24
        # Plot for paper
        ax_ha_image_paper = axarr_final[0]
        ax_150_image_paper = axarr_final[1]
        ax_ha_map_paper = axarr_final[2]
        ax_pab_overlay_paper = axarr_final[3]
        if add_vj_color:
            ax_vj_color_paper = axarr_final[4]

        obj_skycoord = get_coords(id_msa)

        sps_df = read_SPS_cat()
        id_dr3 = sps_df[sps_df['id_msa'] == id_msa]['id_DR3'].iloc[0]

        # Get 150 grayscale
        image_150m, wht_image_150m, photfnu_150m = get_cutout(obj_skycoord, 'f150w', size=image_size)
        # image_410m, wht_image_410m, photfnu_410m = get_cutout(obj_skycoord, 'f410m', size=image_size)
        
        import matplotlib.patheffects as pe

        #Image plots
        cmap_paper = 'inferno'
        ax_ha_image_paper.imshow(ha_image, origin='lower')
        ax_150_image_paper.imshow(image_150m.data, cmap='Greys_r', origin='lower')

        ax_ha_map_paper.imshow(ha_linemap_logscaled, cmap=cmap_paper, norm=ha_linemap_norm, origin='lower')
        ax_pab_overlay_paper.imshow(ha_linemap_logscaled, cmap=cmap_paper, norm=ha_linemap_norm, origin='lower')
        vj_map_norm = Normalize(-0.5, 1.5)
        if add_vj_color:
            ax_vj_color_paper.imshow(vj_map, cmap='Greys_r', norm=vj_map_norm, origin='lower')

        # Get pixesl per 1 kpc for scale
        pix_per_kpc = find_pix_per_kpc(redshift)
        
        axis_x = 0.05
        axis_y = 0.05
        axis_to_data = ax.transAxes + ax.transData.inverted()
        data_x, data_y = axis_to_data.transform((axis_x, axis_y))
        data_x2, data_y2 = axis_to_data.transform((axis_x, axis_y+0.02))
        ax_ha_image_paper.plot([data_x,data_x+(0.5/pixel_scale)], [data_y,data_y], ls='-', color='white', lw=3)
        ax_ha_image_paper.text(data_x, data_y2, '0.5"', color='white', fontsize=14)
        ax_ha_image_paper.text(0.76, 0.04, f'{id_dr3}', fontsize=14, transform=ax_ha_image_paper.transAxes, color='white')

        # Kpc scalebar
        # ax_ha_image_paper.plot([5,5+pix_per_kpc], [10,10], ls='-', color='white', lw=3)
        # ax_ha_image_paper.text(5, 9, '1kpc', color='white')

        # Add filters to HaImage
        text_height = 0.92
        text_start = 0.03
        text_sep = 0.35
        ax_ha_image_paper.text(text_start, text_height, f'{ha_filters[2][2:].upper()}', fontsize=14, transform=ax_ha_image_paper.transAxes, color='blue', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        ax_ha_image_paper.text(text_start+text_sep, text_height, f'{ha_filters[1][2:].upper()}', fontsize=14, transform=ax_ha_image_paper.transAxes, color='green', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        ax_ha_image_paper.text(text_start+2*text_sep, text_height, f'{ha_filters[0][2:].upper()}', fontsize=14, transform=ax_ha_image_paper.transAxes, color='red', bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
        ax_150_image_paper.text(text_start+text_sep, text_height, f'F150W', fontsize=14, transform=ax_150_image_paper.transAxes, color='white', path_effects=[pe.withStroke(linewidth=3, foreground="black")])
        ax_ha_map_paper.text(text_start+text_sep, text_height, f'H$\\alpha$ Map', fontsize=14, transform=ax_ha_map_paper.transAxes, color='white', path_effects=[pe.withStroke(linewidth=3, foreground="black")])
        ax_pab_overlay_paper.text(text_start+text_sep-0.15, text_height, f'H$\\alpha$ Map with Pa$\\beta$', fontsize=14, transform=ax_pab_overlay_paper.transAxes, color='white', path_effects=[pe.withStroke(linewidth=3, foreground="black")])
        

        # PaB Contours
        x = np.arange(pab_linemap.shape[1])
        y = np.arange(pab_linemap.shape[0])
        X_pab, Y_pab = np.meshgrid(x, y)
        # Make a copy of the snr map to be used for contouring
        pab_contour_map = deepcopy(pab_linemap_snr)
        # pab_contour_map[~combined_mask_segmap.mask] = 0
        ax_pab_overlay_paper.contour(X_pab, Y_pab, pab_contour_map, levels=[1,2,3,4,5], cmap='Greys')

        # Ha Contours
        x = np.arange(ha_linemap.shape[1])
        y = np.arange(ha_linemap.shape[0])
        X_ha, Y_ha = np.meshgrid(x, y)
        ha_contour_map = deepcopy(ha_linemap_snr)
        # ha_contour_map[~combined_mask_segmap.mask] = 0   # segmap masking

        if add_vj_color:
            ax_vj_color_paper.text(text_start, text_height, f'{UVJ_filt_names[1].upper()}', fontsize=14, transform=ax_vj_color_paper.transAxes, color='black', path_effects=[pe.withStroke(linewidth=3, foreground="white")])
            ax_vj_color_paper.text(text_start+2*text_sep, text_height, f'{UVJ_filt_names[2].upper()}', fontsize=14, transform=ax_vj_color_paper.transAxes, color='black', path_effects=[pe.withStroke(linewidth=3, foreground="white")])
            ax_vj_color_paper.contour(X_pab, Y_pab, pab_contour_map, levels=[1,2,3,4,5], cmap='Greys')
        
        def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
            new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
            cmap(np.linspace(minval, maxval, n)))
            return new_cmap
        cmap_ha = plt.get_cmap(cmap_paper)
        new_cmap_ha = truncate_colormap(cmap_ha, 0.35, 0.9)
        # if plt_aperture_paper == False: # Remove the contours when showing apertures
        ax_150_image_paper.contour(X_ha, Y_ha, ha_contour_map, levels=[1,2,3,4,5], cmap=new_cmap_ha)

        for ax in axarr_final:
            scale_aspect(ax)
            ax.set_xticks([]); ax.set_yticks([])
        
        ax_ha_image_paper.set_ylabel(label_str, fontsize=paper_font)
        # ax_ha_image_paper.set_xlabel(f'{id_msa}', fontsize=paper_font)

        if plt_aperture_paper:
            # aperture_circle = plt.Circle((image_size[0]/2, image_size[1]/2), aperture/pixel_scale, edgecolor='green', facecolor='None', lw=3)
            # ax_150_image_paper.add_patch(aperture_circle)
            # aperture_circle = plt.Circle((image_size[0]/2, image_size[1]/2), aperture/pixel_scale, edgecolor='green', facecolor='None', lw=3)
            # ax_pab_overlay_paper.add_patch(aperture_circle)
            # plot_shutter_pos(ax_150_image_paper, id_msa, ha_images[1].wcs)
            plot_shutter_pos(ax_ha_image_paper, id_msa, ha_images[1].wcs, paper=True)

        if ax_labels:
            pass
            # ax_ha_image_paper.set_title(f'H$\\alpha$ Image', fontsize=paper_font)
            # ax_150_image_paper.set_title(f'F150W', fontsize=paper_font)
            # ax_ha_map_paper.set_title(f'H$\\alpha$ Linemap', fontsize=paper_font)
            # ax_pab_overlay_paper.set_title(f'Pa$\\beta$ Contours', fontsize=paper_font)
    
        from matplotlib.lines import Line2D
        line_ha_contour = Line2D([0, 1], [0, 1], color=new_cmap_ha(0.7), marker='None', ls='-')
        line_pab_contour = Line2D([0, 1], [0, 1], color='grey', marker='None', ls='-')
        custom_lines_ha = [line_ha_contour]
        custom_labels_ha = ['H$\\alpha$']
        custom_lines_pab = [line_pab_contour]
        custom_labels_pab = ['Pa$\\beta$']
        ax_150_image_paper.legend(custom_lines_ha, custom_labels_ha, loc=3, fontsize=14)
        ax_pab_overlay_paper.legend(custom_lines_pab, custom_labels_pab, loc=3, fontsize=14)

        if plt_aperture_paper:
            # masked_shutter_arr = np.ma.masked_where(combined_shutter_arr < 0.1, combined_shutter_arr)
            # ax_150_image_paper.imshow(masked_shutter_arr, origin='lower')
            pass


    plt.close('all')

    return sed_lineratios, emission_lineratios, sed_avs, emission_avs, err_sed_linefluxes, shutter_calcs, r_value_info, sim_index_values, r_value_info_pabcont_pabline, r_value_info_pabcont_pabline_snrcut


def get_norm(image_map, scalea=1, lower_pct=10, upper_pct=99):
        # imagemap_scaled = np.log(scalea*image_map + 1) / np.log(scalea + 1)  
        # imagemap_scaled = np.emath.logn(1000, image_map)  # = [3, 4] 
        imagemap_gt0 = image_map[image_map>0.0001]
        # imagemap_gt0 = image_map[image_map>0.0001]

        # norm = LogNorm(vmin=np.percentile(imagemap_gt0,lower_pct), vmax=np.percentile(imagemap_gt0,upper_pct))
        norm = Normalize(vmin=np.percentile(imagemap_gt0,lower_pct), vmax=np.percentile(imagemap_gt0,upper_pct))
        return norm

def make_3color(id_msa, line_index = 0, plot = False, image_size=(100,100), paalpha=False, paalpha_pabeta=False): 
    obj_skycoord = get_coords(id_msa)

    line_name = line_list[line_index][0]

    filt_red, filt_green, filt_blue, all_filts = find_filters_around_line(id_msa, line_index, paalpha=paalpha, paalpha_pabeta=paalpha_pabeta)
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
        ax.imshow(image, origin='lower')
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

def get_uvj_images(supercat_df, redshift, id_msa, image_size=(100,100), use_mbands_only=True): # Should I allow wide bands? 
    obj_skycoord = get_coords(id_msa)

    rest_waves = [3650, 5510, 12200] # UVJ filters, angstroms
    observed_waves = [rest_wave * (1+redshift) for rest_wave in rest_waves]

    filt_cols = get_filt_cols(supercat_df)
    uncover_filt_dict, sedpy_filts = unconver_read_filters()
    if use_mbands_only:
        filt_cols = [filt_col for filt_col in filt_cols if 'w' not in filt_col]
    filter_centers = [uncover_filt_dict[filt+'_wave_eff'] for filt in filt_cols]
    nearest_idxs = [min(range(len(filter_centers)), key=lambda i: abs(filter_centers[i]-obs_wave)) for obs_wave in observed_waves]
    filt_names = [filt_cols[idx].split('_')[1] for idx in nearest_idxs]

    image_U, wht_image_U, photfnu_U = get_cutout(obj_skycoord, filt_names[0], size=image_size)
    image_V, wht_image_V, photfnu_V = get_cutout(obj_skycoord, filt_names[1], size=image_size)
    image_J, wht_image_J, photfnu_J = get_cutout(obj_skycoord, filt_names[2], size=image_size)
    images = [image_U, image_V, image_J]
    wht_images = [wht_image_U, wht_image_V, wht_image_J]
    photfnus = [photfnu_U, photfnu_V, photfnu_J]

    return filt_names, images, wht_images, photfnus


def get_coords(id_msa):
    supercat_df = read_supercat()
    row = supercat_df[supercat_df['id_msa']==id_msa]
    if id_msa == 42041:
        row = supercat_df[supercat_df['id'] == 54635]
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


def find_pixels_above_sky_noise(map, segmap_idx, snr_thresh_map=3):
    """Measures sky outside of segmap, then finds the pixels at high enough SNR above it"""
    # Currently removing snr cut
    # sky_pixels = map[~segmap_idx]
    # noise = np.std(sky_pixels)
    # map_highsnr_idxs = map>(snr_thresh_map*noise)
    
    # Make sure it's in the segmap
    map_highsnr_idxs = np.logical_and(map, segmap_idx)
    # map_highsnr_idxs = np.logical_and(map_highsnr_idxs, segmap_idx)
    return map_highsnr_idxs

def plot_and_correlate_highsnr_pix(id_dr3, map1, map_highsnr_idx1, map1_name, map2, map_highsnr_idx2, map2_name, snr_thresh_map, plot=True, bootstrap=0):
    map_both_idxs = np.logical_or(map_highsnr_idx1, map_highsnr_idx2) # can set to OR or AND
    # map_both_idxs = np.logical_or(map_highsnr_idx1, map_highsnr_idx2)
    map1_pixels = map1[map_both_idxs]
    map2_pixels = map2[map_both_idxs]
    if plot:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.plot(map2_pixels, map1_pixels, marker='o', ls='None', color='black')
        ax.set_xscale('log')
        ax.set_yscale('log')
    if len(map1_pixels) < 2 or len(map2_pixels) < 2:
        r_value = -99
        p_value = -99
    else:
        r_value, p_value = pearsonr(map1_pixels, map2_pixels)
    if bootstrap > 0:
        boot_rs = []
        for i in range(bootstrap):
            indices = [random.choice(np.arange(len(map1_pixels))) for k in range(len(map1_pixels))]
            map1_pix_boot = [map1_pixels[k] for k in indices]
            map2_pix_boot = [map2_pixels[k] for k in indices]
            boot_r_value, boot_p_value = pearsonr(map1_pix_boot, map2_pix_boot)
            boot_rs.append(boot_r_value)
    
    n_pixels = len(map1_pixels)
    if plot:
        ax.set_ylabel(map1_name)
        ax.set_xlabel(map2_name)
        ax.set_title(f'r_value = {r_value:0.3f}, p_value = {p_value:0.2e}')
        fig.savefig(f'/Users/brianlorenz/uncover/Figures/dust_map_correlations/{id_dr3}_{map1_name}_{map2_name}_correlation_snr{snr_thresh_map}_OR.pdf', bbox_inches='tight')
    # file = open("/Users/brianlorenz/uncover/Data/generated_tables/r_values.txt", "a")
    # file.write(f"{id_dr3} {r_value} {p_value} {snr_thresh_map}\n")
    # file.close()
    r_value_info = [id_dr3, r_value, p_value, n_pixels, snr_thresh_map]
    if bootstrap > 0:
        r_value_info = [id_dr3, r_value, p_value, n_pixels, snr_thresh_map, boot_rs]
    return r_value_info


def scale_linemap_to_other(map1, map2, segmap_idxs):
    map1_arr = map1[segmap_idxs]
    map2_arr = map2[segmap_idxs]
    
    # Need a filter here? I think we just scale everything though

    a21 = np.sum(map2_arr * map1_arr) / np.sum(map1_arr**2)
    b21 = np.sqrt(np.sum((map2_arr - a21 * map1_arr)**2) / np.sum(map2_arr**2))
    
    map1_scaled = map1*a21

    return map1_scaled, a21, b21

def compute_similarity_linemaps(map1_scaled, map2, segmap_idxs):
    map1_arr = map1_scaled[segmap_idxs]
    map2_arr = map2[segmap_idxs]

    residuals = map1_arr - map2_arr
    squared_res_total = np.sum(residuals**2)

    n_pix = len(map1_arr)
    total_flux = np.sum(map2_arr)

    sim_index = squared_res_total / (n_pix*total_flux)

    return sim_index


def find_filters_around_line(id_msa, line_number, paalpha=False, paalpha_pabeta=False):
    """
    Parameters:
    id_msa (int):
    line_number (int): index of the line number in line-list, should be saved in the same way in zqual_df

    """
    supercat_df = read_supercat()
    filt_names = get_filt_cols(supercat_df, skip_wide_bands=True)
    filt_names.sort()
    if paalpha == True:
        paa_str = '_paa'
    if paalpha_pabeta == True:
        paa_str = '_paa_pab'
    else:
        paa_str = ''
    zqual_detected_df = ascii.read(f'/Users/brianlorenz/uncover/zqual_df_simple{paa_str}.csv').to_pandas()
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
    subtract_filt = 1
    if id_msa in [14573, 19896, 24219, 25558, 32111, 35436] and line_number==1:
        subtract_filt = 2
    filt_blue = filt_names[detected_index-subtract_filt].split('_')[1]
    
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


def plot_sed_around_line(ax, filters, sed_df, spec_df, redshift, line_index, transmissions, id_msa, bootstrap=1000, plt_purple_merged_point=True, show_trasm=False, fluxcal_str=''):
    # Controls for various elements on the plot
    plt_verbose_text = show_trasm
    plt_sed_points = 1
    plt_filter_curves = 1
    plt_spectrum = 1
    plot_vlines = 0

    line_wave_rest = line_list[line_index][1]
    line_wave_obs = (line_wave_rest * (1+redshift))/1e4 # micron
    ax.axvline(line_wave_obs, ls='--', color='green') # observed line, Ha or PaB
    if plot_vlines:
        ax.axvline(1.257*(1+redshift), ls='--', color='magenta') # He II https://www.mpe.mpg.de/ir/ISO/linelists/Hydrogenic.html
        ax.axvline(1.083646*(1+redshift), ls='--', color='magenta') # He I https://iopscience.iop.org/article/10.3847/1538-3881/ab3a31
    ax.axvline(1.094*(1+redshift), ls='--', color='green') # Pa gamma
    # Can check lines here https://linelist.pa.uky.edu/atomic/query.cgi
    
    def set_error_floor(flux, err_flux, floor_pct=0.05):
        err_floor = flux*0.05
        if err_flux < err_floor:
            err_flux = err_floor
        return err_flux

    # Plot the 3 SED points
    for i in range(len(filters)):
        sed_row = sed_df[sed_df['filter'] == filters[i]]
        
        
        if i == 0:
            red_wave = sed_row['eff_wavelength'].iloc[0]
            red_flux = sed_row['flux'].iloc[0]
            err_red_flux = sed_row['err_flux'].iloc[0]
            # err_red_flux = set_error_floor(red_flux, err_red_flux)
        if i == 1:
            green_wave = sed_row['eff_wavelength'].iloc[0]
            green_flux = sed_row['flux'].iloc[0]
            err_green_flux = sed_row['err_flux'].iloc[0]
            # err_green_flux = set_error_floor(green_flux, err_green_flux)
        if i == 2:
            blue_wave = sed_row['eff_wavelength'].iloc[0]
            blue_flux = sed_row['flux'].iloc[0]
            err_blue_flux = sed_row['err_flux'].iloc[0]
            # err_blue_flux = set_error_floor(blue_flux, err_blue_flux)
            # if id_msa == 14573 and line_index==1:
            #     he1_fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting_he1_only/{id_msa}_emission_fits.csv').to_pandas()
            #     he1_flux = he1_fit_df['flux'].iloc[0]
            #     he1_wave = 10830
            #     c = 299792458 # m/s
            #     he1_flux_jy = he1_flux / (1e-23*1e10*c / ((he1_wave)**2))
                
            #     sedpy_blue_name = filters[2].replace('f_', 'jwst_')
            #     sedpy_blue_filt = observate.load_filters([sedpy_blue_name])[0]
            #     blue_filter_width = sedpy_blue_filt.rectangular_width

            #     he1_flux_jy_dispersed = he1_flux_jy / blue_filter_width

            #     blue_flux = blue_flux - he1_flux_jy_dispersed

        # Read and plot each filter curve
        sedpy_name = filters[i].replace('f_', 'jwst_')
        sedpy_filt = observate.load_filters([sedpy_name])[0]
        if plt_filter_curves:
            ax.plot(sedpy_filt.wavelength/1e4, sedpy_filt.transmission/6e5, ls='-', marker='None', color=colors[i], lw=1)

    if plt_sed_points:
        ax.errorbar(red_wave, red_flux, yerr = err_red_flux, color=colors[0], marker='o')
        ax.errorbar(green_wave, green_flux, yerr = err_green_flux, color=colors[1], marker='o')
        ax.errorbar(blue_wave, blue_flux, yerr = err_blue_flux, color=colors[2], marker='o')


    # Compute the percentile to use when combining the continuum
    connect_color = 'purple'
    
    wave_pct = compute_wavelength_pct(blue_wave, green_wave, red_wave)
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
            boot_cont_percentile = compute_cont_pct(blue_wave, green_wave, red_wave, boot_blue_flux, boot_red_flux)
            boot_line, boot_cont = compute_line(boot_cont_percentile, boot_red_flux[0], boot_green_flux[0], boot_blue_flux[0], redshift, 0, filter_width, line_wave_rest)            
            if line_index == 0:
                line_name = 'ha'
            else:
                line_name = 'pab'
            line_trasm = get_line_coverage(id_msa, sedpy_line_filt, redshift, line_name, fluxcal_str=fluxcal_str)
            boot_line = boot_line 
            
            boot_lines.append(boot_line)
    boot_lines = np.array(boot_lines)

    if plt_purple_merged_point:
        ax.plot([red_wave, blue_wave], [red_flux, blue_flux], marker='None', ls='--', color=connect_color)
        ax.plot(green_wave, cont_value, marker='o', ls='None', color=connect_color)
        ax.plot([green_wave,green_wave], [green_flux, cont_value], marker='None', ls='-', color='green', lw=2)
        
        # fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
        # line_flux_fit = fit_df.iloc[line_index]['flux']
        # line_flux_fit_jy = flux_erg_to_jy(line_flux_fit, line_list[line_index][1])

        # ax.text(0.98, 0.85, f'SED: {line_flux:.2e}', color='black', transform=ax.transAxes, horizontalalignment='right')
        # ax.text(0.98, 0.79, f'EmFit: {line_flux_fit_jy:.2e}', color='black', transform=ax.transAxes, horizontalalignment='right')
        # ax.text(0.98, 0.73, f'SED/Fit: {(line_flux/line_flux_fit_jy):.2f}', color='black', transform=ax.transAxes, horizontalalignment='right')

    # Plot the spectrum
    if plt_spectrum:
        ax.step(spec_df['wave'], spec_df['flux_calibrated_jy'], ls='-', marker='None', color='black', lw=1, label='Spectrum')

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
    return cont_percentile, line_flux, trasm_flag, boot_lines, sed_fluxes, wave_pct


def compute_cont_pct(blue_wave, green_wave, red_wave, blue_flux, red_flux):
    total_wave_diff = blue_wave - red_wave
    line_wave_diff = green_wave - red_wave
    cont_percentile = line_wave_diff/total_wave_diff
    if red_flux>blue_flux:
        cont_percentile = 1-cont_percentile
    return cont_percentile

def compute_wavelength_pct(blue_wave, green_wave, red_wave):
    total_wave_diff = red_wave - blue_wave
    line_wave_diff = green_wave - blue_wave
    wave_pct = line_wave_diff/total_wave_diff
    return wave_pct


def compute_line(cont_pct, red_flx, green_flx, blue_flx, redshift, raw_transmission, filter_width, line_rest_wave, images=False, image_noises=[], wave_pct=50):
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

        if images == True:
            err_cont_value = np.sqrt((((wave_pct)**2)*(image_noises[0])**2) + (((1-wave_pct)**2)*(image_noises[2])**2))
            err_line_value = np.sqrt(image_noises[1]**2 + err_cont_value**2)
            err_line_value = err_line_value * 1e-23
            err_line_value = err_line_value * ((c*1e10) / (observed_wave)**2)
            err_line_value = err_line_value * filter_width
            return line_value, cont_value, err_line_value

        # Scale by raw transmission curve
        # line_value = line_value / raw_transmission

        #Scale by transmission
        # line_value = line_value / scaled_transmission
        return line_value, cont_value

def flux_erg_to_jy(line_flux_erg, line_wave):
    c = 299792458 # m/s
    line_flux_jy = line_flux_erg / (1e-23*1e10*c / ((line_wave)**2))
    return line_flux_jy

def get_snr_cut(linemap_snr, snr_thresh=2):
    snr_thresh_line = snr_thresh
    snr_idxs = linemap_snr > snr_thresh_line
    return snr_thresh_line, snr_idxs

def make_combined_mask(snr_binary_map, segmap_idxs):
        combined_mask = np.logical_and(snr_binary_map>0, segmap_idxs)
        total_mask = np.ma.masked_where(combined_mask+1 > 1.5, combined_mask+1)
        return total_mask

def make_all_dustmap(id_msa_list, full_sample=False, fluxcal=True):
    if fluxcal:
        fluxcal_str = ''
    else:
        fluxcal_str = '_no_fluxcal'

    sed_lineratios = []
    sed_lineratios_low = []
    sed_lineratios_high = []
    sed_avs = []
    sed_avs_low = []
    sed_avs_high = []
    emission_lineratios = []
    emission_lineratios_low = []
    emission_lineratios_high = []
    emission_avs = []
    emission_avs_low = []
    emission_avs_high = []
    err_nii_cor_sed_ha_lineflux_lows = []
    err_nii_cor_sed_ha_lineflux_highs = []
    err_fe_cor_sed_pab_lineflux_lows = []
    err_fe_cor_sed_pab_lineflux_highs = []
    err_sed_ha_lineflux_lows = []
    err_sed_ha_lineflux_highs = []

    # Shutter calsc
    ha_shutter_fluxs = []
    pab_shutter_fluxs = []
    lineratio_shutters = []
    av_shutters = []

    # R_vales
    id_dr3s = []
    r_values = []
    p_values = []
    n_pixels = []
    snr_thresh_maps = []
    r_value_16_boots = []
    r_value_84_boots = []
    r_value_16_mcs = []
    r_value_84_mcs = []
    
    # Sim index
    cross_cor_vals = []
    sim_index_vals = []

    # R_vales_pab
    id_dr3s_pab = []
    r_values_pab = []
    p_values_pab = []
    n_pixels_pab = []
    snr_thresh_maps_pab = []
    # R_vales_pab_snr
    id_dr3s_pab_snrcut = []
    r_values_pab_snrcut = []
    p_values_pab_snrcut = []
    n_pixels_pab_snrcut = []
    snr_thresh_maps_pab_snrcut = []
    
    for id_msa in id_msa_list:
        print(f'Making dustmap for {id_msa}')
        sed_lineratios_grouped, emission_lineratios_grouped, sed_avs_grouped, emission_avs_grouped, err_sed_linefluxes_grouped, shutter_calcs, r_value_info, sim_index_values, r_value_info_pabcont_pabline, r_value_info_pabcont_pabline_snrcut = make_dustmap_simple(id_msa, fluxcal_str=fluxcal_str)
        try:
            sed_lineratios_grouped, emission_lineratios_grouped, sed_avs_grouped, emission_avs_grouped, err_sed_linefluxes_grouped, shutter_calcs, r_value_info, sim_index_values, r_value_info_pabcont_pabline, r_value_info_pabcont_pabline_snrcut = make_dustmap_simple(id_msa, fluxcal_str=fluxcal_str)
        except Exception as error:
            print(error)
            print('ERROR')
            print('ERROR - DID NOT COMPUTE - CHECK CODE')
            print('ERROR')
            print('ERROR')
            sed_lineratios_grouped = [-99,-99,-99]
            emission_lineratios_grouped = [-99,-99,-99]
            sed_avs_grouped = [-99,-99,-99]
            emission_avs_grouped = [-99,-99,-99]
            err_sed_linefluxes_grouped = [-99, -99, -99, -99, -99, -99]
            shutter_calcs = [-99, -99, -99, -99]
            r_value_info = [-99, -99, -99, -99, -99, -99, -99, -99, -99]
            sim_index_values = [-99, -99]
            r_value_info_haline_pabline = [-99, -99, -99, -99, -99]
            r_value_info_haline_pabline_snrcut = [-99, -99, -99, -99, -99]
        sed_lineratios.append(sed_lineratios_grouped[0])
        sed_lineratios_low.append(sed_lineratios_grouped[1])
        sed_lineratios_high.append(sed_lineratios_grouped[2])
        sed_avs.append(sed_avs_grouped[0])
        sed_avs_low.append(sed_avs_grouped[1])
        sed_avs_high.append(sed_avs_grouped[2])
        emission_lineratios.append(emission_lineratios_grouped[0])
        emission_lineratios_low.append(emission_lineratios_grouped[1])
        emission_lineratios_high.append(emission_lineratios_grouped[2])
        emission_avs.append(emission_avs_grouped[0])
        emission_avs_low.append(emission_avs_grouped[1])
        emission_avs_high.append(emission_avs_grouped[2])
        err_nii_cor_sed_ha_lineflux_lows.append(err_sed_linefluxes_grouped[0])
        err_nii_cor_sed_ha_lineflux_highs.append(err_sed_linefluxes_grouped[1])
        err_fe_cor_sed_pab_lineflux_lows.append(err_sed_linefluxes_grouped[2])
        err_fe_cor_sed_pab_lineflux_highs.append(err_sed_linefluxes_grouped[3])
        err_sed_ha_lineflux_lows.append(err_sed_linefluxes_grouped[4])
        err_sed_ha_lineflux_highs.append(err_sed_linefluxes_grouped[5])

        ha_shutter_fluxs.append(shutter_calcs[0])
        pab_shutter_fluxs.append(shutter_calcs[1])
        lineratio_shutters.append(shutter_calcs[2])
        av_shutters.append(shutter_calcs[3])

        id_dr3s.append(r_value_info[0])
        r_values.append(r_value_info[1])
        p_values.append(r_value_info[2])
        n_pixels.append(r_value_info[3])
        snr_thresh_maps.append(r_value_info[4])
        r_value_16_boots.append(r_value_info[5])
        r_value_84_boots.append(r_value_info[6])
        r_value_16_mcs.append(r_value_info[7])
        r_value_84_mcs.append(r_value_info[8])

        cross_cor_vals.append(sim_index_values[0])
        sim_index_vals.append(sim_index_values[1])

        id_dr3s_pab.append(r_value_info_pabcont_pabline[0])
        r_values_pab.append(r_value_info_pabcont_pabline[1])
        p_values_pab.append(r_value_info_pabcont_pabline[2])
        n_pixels_pab.append(r_value_info_pabcont_pabline[3])
        snr_thresh_maps_pab.append(r_value_info_pabcont_pabline[4])
        # R_vales_pab_snr
        id_dr3s_pab_snrcut.append(r_value_info_pabcont_pabline_snrcut[0])
        r_values_pab_snrcut.append(r_value_info_pabcont_pabline_snrcut[1])
        p_values_pab_snrcut.append(r_value_info_pabcont_pabline_snrcut[2])
        n_pixels_pab_snrcut.append(r_value_info_pabcont_pabline_snrcut[3])
        snr_thresh_maps_pab_snrcut.append(r_value_info_pabcont_pabline_snrcut[4])





    dustmap_info_df = pd.DataFrame(zip(id_msa_list, sed_lineratios, sed_lineratios_low, sed_lineratios_high, sed_avs, sed_avs_low, sed_avs_high, emission_lineratios, emission_lineratios_low, emission_lineratios_high, emission_avs, emission_avs_low, emission_avs_high, err_nii_cor_sed_ha_lineflux_lows, err_nii_cor_sed_ha_lineflux_highs, err_fe_cor_sed_pab_lineflux_lows, err_fe_cor_sed_pab_lineflux_highs, err_sed_ha_lineflux_lows, err_sed_ha_lineflux_highs), columns=['id_msa', 'sed_lineratio', 'err_sed_lineratio_low', 'err_sed_lineratio_high', 'sed_av', 'err_sed_av_low', 'err_sed_av_high', 'emission_fit_lineratio', 'err_emission_fit_lineratio_low', 'err_emission_fit_lineratio_high', 'emission_fit_av', 'err_emission_fit_av_low', 'err_emission_fit_av_high', 'err_nii_cor_sed_ha_lineflux_low', 'err_nii_cor_sed_ha_lineflux_high', 'err_fe_cor_sed_pab_lineflux_low', 'err_fe_cor_sed_pab_lineflux_high', 'err_sed_ha_lineflux_low', 'err_sed_ha_lineflux_high'])

    shutter_calc_df = pd.DataFrame(zip(id_msa_list, ha_shutter_fluxs, pab_shutter_fluxs, lineratio_shutters, av_shutters), columns=['id_msa', 'ha_shutter_flux', 'pab_shutter_flux', 'lineratio_shutter', 'av_shutter'])

    r_value_df = pd.DataFrame(zip(id_msa_list, id_dr3s, r_values, p_values, n_pixels, snr_thresh_maps, r_value_16_mcs, r_value_84_mcs, r_value_16_boots, r_value_84_boots), columns=['id_msa', 'id_dr3', 'r_value', 'p_value', 'n_pixels', 'snr_thresh_map', 'r_value_16_mc', 'r_value_84_mc', 'r_value_16_boot', 'r_value_84_boot'])
    r_value_df['standard_error'] = (1-r_value_df['r_value']**2) / np.sqrt(r_value_df['n_pixels']-3)

    sim_value_df = pd.DataFrame(zip(id_msa_list, id_dr3s, cross_cor_vals, sim_index_vals), columns=['id_msa', 'id_dr3', 'cross_cor_val', 'sim_index_val'])

    r_value_df_pab = pd.DataFrame(zip(id_msa_list, id_dr3s_pab, r_values_pab, p_values_pab, n_pixels_pab, snr_thresh_maps_pab), columns=['id_msa', 'id_dr3', 'r_value', 'p_value', 'n_pixels', 'snr_thresh_map'])

    r_value_df_pab_snrcut = pd.DataFrame(zip(id_msa_list, id_dr3s_pab_snrcut, r_values_pab_snrcut, p_values_pab_snrcut, n_pixels_pab_snrcut, snr_thresh_maps_pab_snrcut), columns=['id_msa', 'id_dr3', 'r_value', 'p_value', 'n_pixels', 'snr_thresh_map'])

    if full_sample:
        dustmap_info_df.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/lineratio_av_df_all{fluxcal_str}.csv', index=False)
        shutter_calc_df.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/shutter_calcs_all{fluxcal_str}.csv', index=False)
        r_value_df.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/r_values/hacont_haline_r_values_all{fluxcal_str}_snr{int(r_value_df.iloc[0]["snr_thresh_map"])}.csv', index=False)
        sim_value_df.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/r_values/hacont_haline_sim_values_all.csv', index=False)
        r_value_df_pab.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/r_values/pabline_pabcont_r_values_all{fluxcal_str}_snr{int(r_value_df.iloc[0]["snr_thresh_map"])}.csv', index=False)
        r_value_df_pab_snrcut.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/r_values/pabline_pabcont_r_values_all{fluxcal_str}_snr{int(r_value_df.iloc[0]["snr_thresh_map"])}_snrcut.csv', index=False)
    else:
        dustmap_info_df.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/lineratio_av_df{fluxcal_str}.csv', index=False)
        shutter_calc_df.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/shutter_calcs{fluxcal_str}.csv', index=False)
        r_value_df.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/r_values/hacont_haline_r_values{fluxcal_str}_snr{int(r_value_df.iloc[0]["snr_thresh_map"])}.csv', index=False)
        sim_value_df.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/r_values/hacont_haline_sim_values.csv', index=False)
        r_value_df_pab.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/r_values/pabline_pabcont_r_values{fluxcal_str}_snr{int(r_value_df.iloc[0]["snr_thresh_map"])}.csv', index=False)
        r_value_df_pab_snrcut.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/r_values/pabline_pabcont_r_values{fluxcal_str}_snr{int(r_value_df.iloc[0]["snr_thresh_map"])}_snrcut.csv', index=False)

def copy_selected_sample_dustmaps(id_msa_list):
    # Copies all the dustmaps from the sample subset to a different folder
    import os
    import stat
    import shutil
    import time


    dustmap_dir = '/Users/brianlorenz/uncover/Figures/dust_maps/'
    destination = '/Users/brianlorenz/uncover/Figures/dust_maps/sample_only/'

    # List of the paths to all the figures
    fig_list = []
    for id_msa in id_msa_list:
        fig_list.append(dustmap_dir + f'{id_msa}_dustmap.pdf')
    
    def copy_figure(fig_path):
        fig_name = fig_path.split('/')[-1]

        fileStatsObj = os.stat(fig_path)
        modificationTime = time.ctime(fileStatsObj[stat.ST_MTIME])
        print(f"Last Modified {fig_name}: ", modificationTime)

        target = destination + '/' + fig_name
        shutil.copyfile(fig_path, target)


    def copy_all_figures(fig_list):
        for fig_path in fig_list:
            copy_figure(fig_path)

    copy_all_figures(fig_list)

def make_paper_fig_dustmaps(id_msa_list, sortby = 'mass'):
    import math
    rows_per_page = 5
    n_pages = math.ceil(len(id_msa_list) / rows_per_page)
    sps_df = read_SPS_cat()
    lineratio_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineratio_av_df.csv').to_pandas()
    merged_df = lineratio_df.merge(sps_df, on='id_msa')

    if sortby == 'mass':
        merged_df=merged_df.sort_values('mstar_50')
        sort_name = 'mass'
    if sortby == 'av':
        merged_df=merged_df.sort_values('sed_av')
        sort_name = 'av'
    merged_df=merged_df.reset_index(drop=True)

        
    # Need to sort the ids first
    index = 0
    for i in range(n_pages):
        if i == 0:
            rows_this_page = len(id_msa_list) % rows_per_page
            if rows_this_page == 0:
                rows_this_page = rows_per_page
        else:
            rows_this_page = rows_per_page
        n_cols = 4
        if add_vj_color:
            n_cols = n_cols+1
        fig, axarr = plt.subplots(rows_this_page, n_cols, figsize=(4*n_cols,4*rows_this_page))
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
        
        for j in range(rows_this_page):
            if index == len(id_msa_list):
                for ax in axarr[j]:
                    ax.set_axis_off()
                continue
            id_msa = int(merged_df.iloc[index]['id_msa'])
            print(id_msa)
            axarr_id_msa = axarr[j]
            if j == 0:
                ax_labels = True
            else:
                ax_labels = False
            if sortby == 'mass':
                label_str = f"Mass: {merged_df.iloc[index]['mstar_50']:.2f}"
            elif sortby == 'av':
                label_str = f"A$_V$: {merged_df.iloc[index]['sed_av']:.2f}"
            index = index+1
            make_dustmap_simple(id_msa, axarr_final=axarr_id_msa, ax_labels=ax_labels, label_str=label_str)
        fig.savefig(f'/Users/brianlorenz/uncover/Figures/paper_figures/dustmaps_{sort_name}_page{i}.pdf', bbox_inches='tight')

if __name__ == "__main__":
    # make_dustmap_simple(32111)
    # lineflux_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df_all.csv').to_pandas()
    # breakpoint()
    # make_dustmap_simple(47875)
   
    id_msa_list = get_id_msa_list(full_sample=False)
    # make_all_dustmap(id_msa_list, full_sample=False, fluxcal=True)
    make_paper_fig_dustmaps(id_msa_list, sortby='av')
    # make_paper_fig_dustmaps(id_msa_list, sortby='mass')

    # id_msa_list = get_id_msa_list(full_sample=True)
    # make_all_dustmap(id_msa_list, full_sample=True)

    # copy_selected_sample_dustmaps(id_msa_list)
