from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits, ascii
from astropy import units as u
from astropy.nddata import Cutout2D
from astropy.convolution import Gaussian2DKernel, convolve
from uncover_read_data import read_supercat, read_raw_spec, read_spec_cat, read_segmap, read_SPS_cat
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


correct_pab = 1

colors = ['red', 'green', 'blue']
connect_color = 'green'


# class galaxy:
#     def __init__(self, id_msa):
#         self.id_msa = id_msa
#         self.setup_filters()
#         self.read_emission()

#     def setup_filters(self):
#         self.ha_filters, self.ha_images, self.wht_ha_images, self.obj_segmap = make_3color(self.id_msa, line_index=0, plot=False)
#         self.pab_filters, self.pab_images, self.wht_pab_images, self.obj_segmap = make_3color(self.id_msa, line_index=1, plot=False)
#         self.ha_sedpy_name = self.ha_filters[1].replace('f', 'jwst_f')
#         self.ha_sedpy_filt = observate.load_filters([self.ha_sedpy_name])[0]
#         self.pab_sedpy_name = self.pab_filters[1].replace('f', 'jwst_f')
#         self.pab_sedpy_filt = observate.load_filters([self.pab_sedpy_name])[0]

#         self.ha_red_sedpy_name = self.ha_filters[0].replace('f', 'jwst_f')
#         self.ha_red_sedpy_filt = observate.load_filters([self.ha_red_sedpy_name])[0]
#         self.pab_red_sedpy_name = self.pab_filters[0].replace('f', 'jwst_f')
#         self.pab_red_sedpy_filt = observate.load_filters([self.pab_red_sedpy_name])[0]
#         self.ha_blue_sedpy_name = self.ha_filters[2].replace('f', 'jwst_f')
#         self.ha_blue_sedpy_filt = observate.load_filters([self.ha_blue_sedpy_name])[0]
#         self.pab_blue_sedpy_name = self.pab_filters[2].replace('f', 'jwst_f')
#         self.pab_blue_sedpy_filt = observate.load_filters([self.pab_blue_sedpy_name])[0]

#         self.ha_filters = ['f_'+filt for filt in self.ha_filters]
#         self.pab_filters = ['f_'+filt for filt in self.pab_filters]
#         self.spec_df = read_raw_spec(self.id_msa)
#         self.sed_df = read_sed(self.id_msa)
#         zqual_df = read_spec_cat()
#         self.redshift = zqual_df[zqual_df['id_msa']==self.id_msa]['z_spec'].iloc[0]

#         self.ha_line_scaled_transmission = get_transmission_at_line(self.ha_sedpy_filt, line_list[0][1] * (1+self.redshift))
#         self.pab_line_scaled_transmission = get_transmission_at_line(self.pab_sedpy_filt, line_list[1][1] * (1+self.redshift))
#         self.correction_ratio = self.pab_line_scaled_transmission/self.ha_line_scaled_transmission

#     def read_emission(self):
#         self.fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{self.id_msa}_emission_fits.csv').to_pandas()
#         self.line_ratio_from_emission = self.fit_df["ha_pab_ratio"].iloc[0]
#         err_line_ratio_from_emission_low = self.fit_df["err_ha_pab_ratio_low"].iloc[0]
#         err_line_ratio_from_emission_high = self.fit_df["err_ha_pab_ratio_high"].iloc[0]
#         self.emission_lineratios = [self.line_ratio_from_emission, err_line_ratio_from_emission_low, err_line_ratio_from_emission_high]



def make_all_dustmap():
    zqual_df_cont_covered = ascii.read('/Users/brianlorenz/uncover/zqual_df_cont_covered.csv').to_pandas()
    id_msa_list = zqual_df_cont_covered['id_msa']
    spec_ratios = []
    sed_ratios = []
    err_sed_ratios_low = []
    err_sed_ratios_high = []
    emission_ratios = []
    err_emission_ratios_low = []
    err_emission_ratios_high = []
    ha_trasm_flags = []
    pab_trasm_flags = []
    int_spec_ha_compares = []
    int_spec_pab_compares = []
    ha_sed_value_compares = []
    pab_sed_value_compares = []
    err_ha_sed_value_compare_lows = []
    err_ha_sed_value_compare_highs = []
    err_pab_sed_value_compare_lows = []
    err_pab_sed_value_compare_highs = []
    line_ratio_from_spec_fit_sed_prospects = []
    spec_scale_factors = []

    ha_sed_fluxes = []
    pab_sed_fluxes = []
    ha_emfit_fluxes = []
    pab_emfit_fluxes = []
    for id_msa in id_msa_list:
        sed_lineratio, err_sed_lineratios, line_ratio_from_spec, emission_lineratios, ha_trasm_flag, pab_trasm_flag, sed_intspec_compare_values, line_ratio_from_spec_fit_sed_prospect, spec_scale_factor, line_flux_compares = make_dustmap(id_msa)
        sed_ratios.append(sed_lineratio)
        err_sed_ratios_low.append(err_sed_lineratios[0])
        err_sed_ratios_high.append(err_sed_lineratios[1])
        spec_ratios.append(line_ratio_from_spec)
        emission_ratios.append(emission_lineratios[0])
        err_emission_ratios_low.append(emission_lineratios[1])
        err_emission_ratios_high.append(emission_lineratios[2])
        ha_trasm_flags.append(ha_trasm_flag)
        pab_trasm_flags.append(pab_trasm_flag)
        int_spec_ha_compares.append(sed_intspec_compare_values[0])
        int_spec_pab_compares.append(sed_intspec_compare_values[1])
        ha_sed_value_compares.append(sed_intspec_compare_values[2])
        pab_sed_value_compares.append(sed_intspec_compare_values[3])
        err_ha_sed_value_compare_lows.append(sed_intspec_compare_values[4])
        err_ha_sed_value_compare_highs.append(sed_intspec_compare_values[5])
        err_pab_sed_value_compare_lows.append(sed_intspec_compare_values[6])
        err_pab_sed_value_compare_highs.append(sed_intspec_compare_values[7])
        line_ratio_from_spec_fit_sed_prospects.append(line_ratio_from_spec_fit_sed_prospect)
        spec_scale_factors.append(spec_scale_factor)

        ha_sed_fluxes.append(line_flux_compares[0])
        pab_sed_fluxes.append(line_flux_compares[1])
        ha_emfit_fluxes.append(line_flux_compares[2])
        pab_emfit_fluxes.append(line_flux_compares[3])

    compare_emfit_df = pd.DataFrame(zip(id_msa_list, ha_sed_fluxes, pab_sed_fluxes, ha_emfit_fluxes, pab_emfit_fluxes), columns=['id_msa', 'ha_sed_flux', 'pab_sed_flux', 'ha_emfit_flux', 'pab_emfit_flux'])
    compare_emfit_df.to_csv('/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/compare_emfit_df.csv', index=False)

    lineratio_df = pd.DataFrame(zip(id_msa_list, sed_ratios, err_sed_ratios_low, err_sed_ratios_high, spec_ratios, emission_ratios, err_emission_ratios_low, err_emission_ratios_high, int_spec_ha_compares, int_spec_pab_compares, ha_sed_value_compares, pab_sed_value_compares, err_ha_sed_value_compare_lows, err_ha_sed_value_compare_highs, err_pab_sed_value_compare_lows, err_pab_sed_value_compare_highs, line_ratio_from_spec_fit_sed_prospects, spec_scale_factors), columns=['id_msa', 'sed_lineratio', 'sed_lineratio_16', 'sed_lineratio_84', 'integrated_spec_lineratio', 'emission_fit_lineratio', 'err_emission_fit_lineratio_low', 'err_emission_fit_lineratio_high', 'int_spec_ha_compare', 'int_spec_pab_compare', 'sed_ha_compare', 'sed_pab_compare', 'sed_ha_compare_16', 'sed_ha_compare_84', 'sed_pab_compare_16', 'sed_pab_compare_84', 'line_ratio_prospector_fit', 'spec_scale_factor'])
    lineratio_df.to_csv('/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/lineratio_df.csv', index=False)

    
    # Then run "generate filtered line ratio df" from av_compare_figure
    zqual_df_cont_covered['ha_trasm_flag'] = ha_trasm_flags
    zqual_df_cont_covered['pab_trasm_flag'] = pab_trasm_flags
    zqual_df_cont_covered.to_csv('/Users/brianlorenz/uncover/zqual_df_cont_covered.csv', index=False)


def make_all_3color(id_msa_list):
    for id_msa in id_msa_list:
        make_3color(id_msa, line_index=0, plot=True)
        make_3color(id_msa, line_index=1, plot=True)

def make_dustmap(id_msa):
    # Read in the images
    ha_filters, ha_images, wht_ha_images, obj_segmap, ha_photfnus = make_3color(id_msa, line_index=0, plot=False)
    pab_filters, pab_images, wht_pab_images, obj_segmap, pab_photfnus = make_3color(id_msa, line_index=1, plot=False)
    ha_sedpy_name = ha_filters[1].replace('f', 'jwst_f')
    ha_sedpy_filt = observate.load_filters([ha_sedpy_name])[0]
    pab_sedpy_name = pab_filters[1].replace('f', 'jwst_f')
    pab_sedpy_filt = observate.load_filters([pab_sedpy_name])[0]

    ha_red_sedpy_name = ha_filters[0].replace('f', 'jwst_f')
    ha_red_sedpy_filt = observate.load_filters([ha_red_sedpy_name])[0]
    pab_red_sedpy_name = pab_filters[0].replace('f', 'jwst_f')
    pab_red_sedpy_filt = observate.load_filters([pab_red_sedpy_name])[0]
    ha_blue_sedpy_name = ha_filters[2].replace('f', 'jwst_f')
    ha_blue_sedpy_filt = observate.load_filters([ha_blue_sedpy_name])[0]
    pab_blue_sedpy_name = pab_filters[2].replace('f', 'jwst_f')
    pab_blue_sedpy_filt = observate.load_filters([pab_blue_sedpy_name])[0]


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
    sed_df = read_sed(id_msa)
    zqual_df = read_spec_cat()
    redshift = zqual_df[zqual_df['id_msa']==id_msa]['z_spec'].iloc[0]

    ha_line_scaled_transmission = get_transmission_at_line(ha_sedpy_filt, line_list[0][1] * (1+redshift))
    pab_line_scaled_transmission = get_transmission_at_line(pab_sedpy_filt, line_list[1][1] * (1+redshift))
    # correction_ratio = pab_line_scaled_transmission/ha_line_scaled_transmission
    line_transmissions = [ha_line_scaled_transmission, pab_line_scaled_transmission]
    
    
    fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
    line_ratio_from_emission = fit_df["ha_pab_ratio"].iloc[0]
    err_line_ratio_from_emission_low = fit_df["err_ha_pab_ratio_low"].iloc[0]
    err_line_ratio_from_emission_high = fit_df["err_ha_pab_ratio_high"].iloc[0]
    emission_lineratios = [line_ratio_from_emission, err_line_ratio_from_emission_low, err_line_ratio_from_emission_high]
    ha_flux_fit = fit_df.iloc[0]['flux']
    pab_flux_fit = fit_df.iloc[1]['flux']
    ha_sigma = fit_df.iloc[0]['sigma'] # full width of the line
    pab_sigma = fit_df.iloc[1]['sigma'] # full width of the line

    # Check the coverage fraction of the lines - we want it high in the line, but 0 int he continuum filters
    ha_avg_transmission = get_line_coverage(ha_sedpy_filt, line_list[0][1] * (1+redshift), ha_sigma * (1+redshift))
    pab_avg_transmission = get_line_coverage(pab_sedpy_filt, line_list[1][1] * (1+redshift), pab_sigma * (1+redshift))
    ha_red_avg_transmission = get_line_coverage(ha_red_sedpy_filt, line_list[0][1] * (1+redshift), ha_sigma * (1+redshift))
    pab_red_avg_transmission = get_line_coverage(pab_red_sedpy_filt, line_list[1][1] * (1+redshift), pab_sigma * (1+redshift))
    ha_blue_avg_transmission = get_line_coverage(ha_blue_sedpy_filt, line_list[0][1] * (1+redshift), ha_sigma * (1+redshift))
    pab_blue_avg_transmission = get_line_coverage(pab_blue_sedpy_filt, line_list[1][1] * (1+redshift), pab_sigma * (1+redshift))
    ha_transmissions = [ha_red_avg_transmission, ha_avg_transmission, ha_blue_avg_transmission]
    pab_transmissions = [pab_red_avg_transmission, pab_avg_transmission, pab_blue_avg_transmission]
    

    # Segmap matching
    supercat_df = read_supercat()
    id_dr3 = supercat_df[supercat_df['id_msa']==id_msa]['id'].iloc[0]
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
    
    
    
    
    def get_dustmap(halpha_map, pabeta_map): # Ha should be 18 times larger than pab, but it's only 3. Leading to huge Avs
        # Why scaling them first? No need, right?
        # ha_map_scaled = halpha_map/ha_factor
        # pab_map_scaled = pabeta_map/pab_factor
        # dustmap = pab_map_scaled / ha_map_scaled
        # dustmap2 = (halpha_map / pabeta_map) / ((line_list[0][1] / line_list[1][1])**2)
        dustmap = pabeta_map / halpha_map / ((line_list[1][1] / line_list[0][1])**2)
        
        av_dustmap = compute_ha_pab_av_from_dustmap(dustmap)
        # breakpoint()
        
        return dustmap

    
    # Make SED plot, return percentile of line between the other two filters
    ha_cont_pct, ha_sed_lineflux, ha_sed_value_scaled, ha_trasm_flag, ha_boot_lines, ha_green_flux_sed = plot_sed_around_line(ax_ha_sed, ha_filters, sed_df, spec_df, redshift, 0, line_transmissions[0], ha_transmissions, id_msa)
    pab_cont_pct, pab_sed_lineflux, pab_sed_value_scaled, pab_trasm_flag, pab_boot_lines, pab_green_flux_sed = plot_sed_around_line(ax_pab_sed, pab_filters, sed_df, spec_df, redshift, 1, line_transmissions[1], pab_transmissions, id_msa)
    if correct_pab:
        pab_sed_lineflux = pab_sed_lineflux * 0.8
   
    sed_lineratio = compute_lineratio(ha_sed_lineflux, pab_sed_lineflux)
    boot_sed_lineratios = compute_lineratio(ha_boot_lines, pab_boot_lines)
    err_sed_lineratio_low = np.percentile(boot_sed_lineratios, 16)
    err_sed_lineratio_high = np.percentile(boot_sed_lineratios, 84)
    err_sed_lineratios = [err_sed_lineratio_low, err_sed_lineratio_high]
    # sed_lineratio_scaled = compute_lineratio(ha_sed_value_scaled, pab_sed_value_scaled, ha_line_scaled_transmission, pab_line_scaled_transmission)

    line_ratio_from_spec, int_spec_ha, int_spec_pab, line_ratio_from_spec_fit, line_ratio_from_spec_fit_sed, line_ratio_from_spec_fit_sed_prospect = check_line_ratio_spectra(ha_filters, pab_filters, spec_df, sed_df, id_msa, redshift, ax_ha_sed, ax_pab_sed, ha_green_flux_sed, pab_green_flux_sed, sed_lineratio, ha_transmissions, pab_transmissions, line_transmissions)
    print(f'Line ratio from integrated spectrum: {line_ratio_from_spec}')
    print(f'Line ratio from integrated spectrum polyfit: {line_ratio_from_spec_fit}')
    print(f'Line ratio from integrated spectrum polyfit using sed point: {line_ratio_from_spec_fit_sed}')

    # Compare sed and int spec measurements
    
    spec_scale_factor = np.nanmedian(spec_df['scaled_flux'] / spec_df['flux'])
    int_spec_ha_compare = (int_spec_ha * spec_scale_factor) 
    int_spec_pab_compare = (int_spec_pab * spec_scale_factor) 
    ha_sed_value_compare = ha_sed_lineflux 
    pab_sed_value_compare = pab_sed_lineflux 
    err_sed_ha_sed_value_compare_low = np.percentile(ha_boot_lines, 16) 
    err_sed_ha_sed_value_compare_high = np.percentile(ha_boot_lines, 84) 
    err_sed_pab_sed_value_compare_low = np.percentile(pab_boot_lines, 16) 
    err_sed_pab_sed_value_compare_high = np.percentile(pab_boot_lines, 84) 
    sed_intspec_compare_values = [int_spec_ha_compare, int_spec_pab_compare, ha_sed_value_compare, pab_sed_value_compare, err_sed_ha_sed_value_compare_low, err_sed_ha_sed_value_compare_high, err_sed_pab_sed_value_compare_low, err_sed_pab_sed_value_compare_high]

    # Make linemaps
    ha_cont, ha_linemap, ha_image, ha_linemap_snr = get_cont_and_map(ha_images, wht_ha_images, ha_cont_pct, redshift, ha_line_scaled_transmission)
    pab_cont, pab_linemap, pab_image, pab_linemap_snr = get_cont_and_map(pab_images, wht_pab_images, pab_cont_pct, redshift, pab_line_scaled_transmission)
    from copy import copy, deepcopy
    ha_linemap_snr_static = deepcopy(ha_linemap_snr)
    
    # Make dustmap
    dustmap = get_dustmap(ha_linemap, pab_linemap)
    avg_ha_map = np.mean(ha_linemap[48:52,48:52])
    avg_pab_map = np.mean(pab_linemap[48:52,48:52])
    # breakpoint()

    # breakpoint()




    # Set negative points to nonzero values, we take logs during normalization. All calculations are complete by now
    # ha_cont[ha_cont<0] = 0.00001
    # pab_cont[pab_cont<0] = 0.00001
    # ha_linemap[ha_linemap<0] = 0.00001
    # pab_linemap[pab_linemap<0] = 0.00001
    dustmap[dustmap<0.00001] = 0.00001
    dustmap[dustmap>200] = 200
    # Anywhere that halpha was not detected but pabeta was detected, set the dustmap to a high value
    def set_dustmap_av(dustmap, ha_linemap, ha_linemap_snr, pab_linemap, pab_linemap_snr):
        ha_nondetect_idx = ha_linemap<0
        pab_detect_idx = pab_linemap_snr>0.5
        both_idx = np.logical_and(ha_nondetect_idx, pab_detect_idx)
        dustmap[both_idx] = 20
        return dustmap
    dustmap = set_dustmap_av(dustmap, ha_linemap, ha_linemap_snr, pab_linemap, pab_linemap_snr)

    
    
    # SHR calculations, need to check these
    # ax_segmap.imshow(segmap_idxs)
    def get_snr_cut(linemap_snr, snr_thresh=80):
        # snr_thresh_line = np.percentile(linemap_snr, snr_thresh)
        snr_thresh_line = 2
        snr_idxs = linemap_snr > snr_thresh_line
        return snr_thresh_line, snr_idxs
    ha_snr_thresh, ha_snr_idxs = get_snr_cut(ha_linemap_snr)
    pab_snr_thresh, pab_snr_idxs = get_snr_cut(pab_linemap_snr)
    snr_idx = np.logical_or(ha_snr_idxs, pab_snr_idxs)
    snr_idx = ha_snr_maps[1] > np.percentile(ha_snr_maps[1], 75)
    
    ha_snr_idx = ha_snr_maps[1] > 1.5
    pab_snr_idx = pab_snr_maps[1] > 2
    snr_idx = np.logical_and(ha_snr_idx, pab_snr_idx)



    ha_linemap_snr[snr_idx] = 1
    ha_linemap_snr[~snr_idx] = 0

    # snr_idx = np.logical_or(ha_snr_idxs, pab_snr_idxs)
    # snr_idx = pab_snr_maps[1] > np.percentile(pab_snr_maps[1], 70)
    # pab_linemap_snr[snr_idx] = 1
    # pab_linemap_snr[~snr_idx] = 0
    # dustmap[~snr_idx]=0

    
    # Compare to emission fit
    ha_flux_fit_jy = flux_erg_to_jy(ha_flux_fit, line_list[0][1])
    pab_flux_fit_jy = flux_erg_to_jy(pab_flux_fit, line_list[1][1])
    print(ha_sed_lineflux / ha_flux_fit_jy)
    print(pab_sed_lineflux / pab_flux_fit_jy)
    line_flux_compares = [ha_sed_lineflux, pab_sed_lineflux, ha_flux_fit_jy, pab_flux_fit_jy]

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
    dustmap_lower_pct = 40
    dustmap_upper_pct = 90
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

    # Smooth the dust map
    sigma = 3.0  # Standard deviation for Gaussian kernel
    kernel = Gaussian2DKernel(sigma)
    smoothed_dustmap = convolve(dustmap_logscaled, kernel)

    # Showdustmap, masked points in gray
    ax_dustmap.imshow(dustmap_logscaled, cmap=cmap, norm=dustmap_norm)
    masked_dustmap = np.ma.masked_where(ha_linemap_snr+1 > 1.5, ha_linemap_snr+1)
    from matplotlib import colors
    cmap_gray = colors.ListedColormap(['gray'])
    ax_dustmap.imshow(masked_dustmap, cmap=cmap_gray)
    ax_ha_linemap.imshow(masked_dustmap, cmap=cmap_gray)
    ax_pab_linemap.imshow(masked_dustmap, cmap=cmap_gray)
    # breakpoint()

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
        # ax.contour(eroded_segmap_idxs, levels=[0.5], colors='white')


    ax_ha_sed.text(0.50, 1.15, f'z = {round(redshift,2)}', color='black', fontsize=18, transform=ax_ha_sed.transAxes)
    ax_ha_sed.text(-0.05, 1.15, f'id = {id_msa}', color='black', fontsize=18, transform=ax_ha_sed.transAxes)
    ax_ha_image.text(-0.05, 1.10, f'Line ratio from: Int. spectrum: {round(line_ratio_from_spec, 2)}', fontsize=14, transform=ax_ha_image.transAxes)
    ax_ha_image.text(2.2, 1.18, f'Prospectorfit SED point: {round(line_ratio_from_spec_fit_sed_prospect, 2)}', fontsize=14, transform=ax_ha_image.transAxes, color='magenta')
    ax_ha_image.text(1.3, 1.10, f'Polyfit SED point: {round(line_ratio_from_spec_fit_sed, 2)}', fontsize=14, transform=ax_ha_image.transAxes, color='orange')
    ax_ha_image.text(1.3, 1.18, f'Int. spec polyfit: {round(line_ratio_from_spec_fit, 2)}', fontsize=14, transform=ax_ha_image.transAxes, color='lime')
    ax_ha_image.text(2.2, 1.10, f'Emission fit: {round(line_ratio_from_emission, 2)}', fontsize=14, transform=ax_ha_image.transAxes)
    ax_ha_image.text(2.9, 1.10, f'sed: {round(sed_lineratio, 2)}', fontsize=14, transform=ax_ha_image.transAxes, color='purple')
    ax_ha_image.text(3.5, 1.10, f'Fit AV: {round((1/av_lineratio), 2)}', fontsize=14, transform=ax_ha_image.transAxes)
    ax_segmap.text(-0.25, -0.15, f'Ha sigma: {round((ha_sigma), 2)}', fontsize=14, transform=ax_segmap.transAxes)
    ax_segmap.text(0.5, -0.15, f'PaB sigma: {round((pab_sigma), 2)}', fontsize=14, transform=ax_segmap.transAxes)

    # Save
    for ax in ax_list:
        scale_aspect(ax)
    save_folder = '/Users/brianlorenz/uncover/Figures/dust_maps'
    fig.savefig(save_folder + f'/{id_msa}_dustmap.pdf')
    plt.close('all')

    # SEcond figure, zoom-in on the center of the dustmap with ratios
    fig2 = plt.figure(figsize=(14, 8))
    gs2 = GridSpec(2, 3, left=0.05, right=0.99, bottom=0.1, top=0.90, wspace=0.01, hspace=0.3)
    ax_ha_pab = fig2.add_subplot(gs2[0, 2])
    ax_dustmap2 = fig2.add_subplot(gs2[1, 2])
    ax_ha_line = fig2.add_subplot(gs2[0, 0])
    ax_pab_line = fig2.add_subplot(gs2[1, 0])
    ax_ha_snr = fig2.add_subplot(gs2[0, 1])
    ax_pab_snr = fig2.add_subplot(gs2[1, 1])

    zoom_region_min = 45
    zoom_region_max = 55

    av_dustmap = compute_ha_pab_av_from_dustmap(dustmap)

    ha_div_pab = (ha_linemap / pab_linemap) / ((line_list[0][1] / line_list[1][1])**2)
    ax_ha_snr.imshow(ha_linemap_snr_static[zoom_region_min:zoom_region_max, zoom_region_min:zoom_region_max])
    ax_pab_snr.imshow(pab_linemap_snr[zoom_region_min:zoom_region_max, zoom_region_min:zoom_region_max])
    ax_ha_line.imshow(ha_linemap[zoom_region_min:zoom_region_max, zoom_region_min:zoom_region_max])
    ax_pab_line.imshow(pab_linemap[zoom_region_min:zoom_region_max, zoom_region_min:zoom_region_max])
    c_dustmap = ax_dustmap2.imshow(av_dustmap[zoom_region_min:zoom_region_max, zoom_region_min:zoom_region_max], vmin=0, vmax=4)
    c_image = ax_ha_pab.imshow(ha_div_pab[zoom_region_min:zoom_region_max, zoom_region_min:zoom_region_max], cmap=cmap, vmin=0, vmax=30)
    cb = plt.colorbar(c_image,ax=ax_ha_pab)
    cb_dustmap = plt.colorbar(c_dustmap,ax=ax_dustmap2)
    
    ax_ha_snr.set_title('Ha SNR')
    ax_ha_line.set_title('Ha linemap')
    ax_pab_snr.set_title('PaB SNR')
    ax_pab_line.set_title('PaB linemap')
    ax_ha_pab.set_title('Ha / PaB')
    ax_dustmap2.set_title('Dust Map (AV)')
    
    plt.show()


    fig2.savefig(f'/Users/brianlorenz/uncover/Figures/dust_map_zoom/{id_msa}_dust_map_zoom.pdf')

    return sed_lineratio, err_sed_lineratios, line_ratio_from_spec, emission_lineratios, ha_trasm_flag, pab_trasm_flag, sed_intspec_compare_values, line_ratio_from_spec_fit_sed_prospect, spec_scale_factor, line_flux_compares

def plot_sed_around_line(ax, filters, sed_df, spec_df, redshift, line_index, line_scaled_transmission, transmissions, id_msa, bootstrap=1000, plt_purple_merged_point = 0, plt_prospect=0, show_trasm=0):
    # Controls for various elements on the plot
    plt_verbose_text = show_trasm
    plt_sed_points = 1
    plt_filter_curves = 1
    plt_spectrum = 1
    plt_prospector = plt_prospect

    prospector_spec_df, prospector_sed_df = read_prospector(id_msa)

    line_wave_obs = (line_list[line_index][1] * (1+redshift))/1e4
    ax.axvline(line_wave_obs, ls='--', color='green') # Pab
    ax.axvline(1.2560034*(1+redshift), ls='--', color='magenta') # He II https://www.mpe.mpg.de/ir/ISO/linelists/Hydrogenic.html
    ax.axvline(1.083646*(1+redshift), ls='--', color='magenta') # He I https://iopscience.iop.org/article/10.3847/1538-3881/ab3a31
    ax.axvline(1.094*(1+redshift), ls='--', color='green') # Pa gamma
    # Can check lines here https://linelist.pa.uky.edu/atomic/query.cgi
    
    
    # Plot the 3 SED points
    for i in range(len(filters)):
        sed_row = sed_df[sed_df['filter'] == filters[i]]
        if plt_sed_points:
            ax.errorbar(sed_row['eff_wavelength'], sed_row['spec_scaled_flux'], yerr = sed_row['err_flux'], color=colors[i], marker='o')
        
        if i == 0:
            red_wave = sed_row['eff_wavelength'].iloc[0]
            red_flux = sed_row['spec_scaled_flux'].iloc[0]
            err_red_flux = sed_row['err_spec_scaled_flux'].iloc[0]
            red_flux_scaled = red_flux/sed_row['eff_width'].iloc[0]
            if correct_pab == 1 and line_index == 1:
                red_flux = sed_row['spec_scaled_flux'].iloc[0] * 1.04
                err_red_flux = sed_row['err_spec_scaled_flux'].iloc[0] * 1.04
        if i == 1:
            green_wave = sed_row['eff_wavelength'].iloc[0]
            green_flux = sed_row['spec_scaled_flux'].iloc[0]
            err_green_flux = sed_row['err_spec_scaled_flux'].iloc[0]
            green_flux_scaled = green_flux/sed_row['eff_width'].iloc[0]
        if i == 2:
            blue_wave = sed_row['eff_wavelength'].iloc[0]
            blue_flux = sed_row['spec_scaled_flux'].iloc[0]
            err_blue_flux = sed_row['err_spec_scaled_flux'].iloc[0]
            blue_flux_scaled = blue_flux/sed_row['eff_width'].iloc[0]

        # Read and plot each filter curve
        sedpy_name = filters[i].replace('f_', 'jwst_')
        sedpy_filt = observate.load_filters([sedpy_name])[0]
        if plt_filter_curves:
            ax.plot(sedpy_filt.wavelength/1e4, sedpy_filt.transmission/1e6, ls='-', marker='None', color=colors[i], lw=1)
    
   
    # Compute the percentile to use when combining the continuum
    connect_color = 'purple'
    
    cont_percentile = compute_cont_pct(blue_wave, green_wave, red_wave, blue_flux, red_flux)
    def compute_line(cont_pct, red_flx, green_flx, blue_flx, redshift, line_scaled_transmission):
        cont_value = np.percentile([blue_flx, red_flx], cont_pct*100)
        line_value = green_flx - cont_value
        line_flux = line_value / (1+redshift)**2
        line_flux_cor = line_flux / line_scaled_transmission
        return line_flux_cor, cont_value
    line_flux, cont_value = compute_line(cont_percentile, red_flux, green_flux, blue_flux, redshift, line_scaled_transmission)

    boot_lines = []
    if bootstrap > 0:
        for i in range(bootstrap):
            # Remake fluxes:
            if err_red_flux < 0:
                print('NEGATIVE ERROR- NEED TO FIX')
                err_red_flux = np.abs(err_red_flux)
            if err_green_flux < 0:
                print('NEGATIVE ERROR- NEED TO FIX')
                err_green_flux = np.abs(err_green_flux)
            if err_blue_flux < 0:
                print('NEGATIVE ERROR- NEED TO FIX')
                err_blue_flux = np.abs(err_blue_flux)
            boot_red_flux = np.random.normal(loc=red_flux, scale=err_red_flux, size=1)
            boot_green_flux = np.random.normal(loc=green_flux, scale=err_green_flux, size=1)
            boot_blue_flux = np.random.normal(loc=blue_flux, scale=err_blue_flux, size=1)
            boot_line, boot_cont = compute_line(cont_percentile, boot_red_flux, boot_green_flux, boot_blue_flux, redshift, line_scaled_transmission)            
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

    # Compute the percentile to use when combining the continuum
    cont_value_scaled = np.percentile([red_flux_scaled, blue_flux_scaled], cont_percentile*100)
    line_value_scaled = green_flux_scaled - cont_value_scaled

    # Plot the spectrum
    if plt_spectrum:
        ax.plot(spec_df['wave'], spec_df['flux'], ls='-', marker='None', color='black', lw=1, label='Spectrum')

    # Plot the prospector spectrum
    if plt_prospector:
        ax.plot(prospector_spec_df['wave_um'], prospector_spec_df['spec_scaled_flux'], ls='-', marker='None', color='orange', lw=1, label='Prospector Spec')
        ax.plot(prospector_sed_df['weff_um'], prospector_sed_df['spec_scaled_flux'], ls='None', marker='o', color='magenta', lw=1, label='Prospector Sed', zorder=10000000, alpha=0.5)

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
    ax.set_ylim(0, 1.2*np.max(spec_df['flux']))
    return cont_percentile, line_flux, line_value_scaled, trasm_flag, boot_lines, green_flux

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
    
    return filters, images, wht_images, obj_segmap, photfnus

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

def check_line_ratio_spectra(ha_filters, pab_filters, spec_df, sed_df, id_msa, redshift, ax_ha_main, ax_pab_main, ha_green_flux_sed, pab_green_flux_sed, sed_lineratio, ha_transmissions, pab_transmissions, line_transmissions):
    """Measure the line ratio just from integrating spectrum over the trasnmission curve"""
    wavelength = spec_df['wave_aa'].to_numpy()
    f_lambda = spec_df['flux_erg_aa'].to_numpy()
    filt_dict, filters = unconver_read_filters()
    filter_names = [sedpy_filt.name for sedpy_filt in filters]
    integrated_sed_abmag = observate.getSED(wavelength, f_lambda, filterlist=filters)
    integrated_sed_jy = 10**(-0.4*(integrated_sed_abmag-8.9))
    effective_waves_aa = sed_df['eff_wavelength']*10000
    
    ha_idxs = []
    pab_idxs = []
    for ha_filt in ha_filters:
        ha_filt = ha_filt[2:]
        for index, item in enumerate(filter_names):
            if ha_filt in item:
                ha_idxs.append(index)
    for pab_filt in pab_filters:
        pab_filt = pab_filt[2:]
        for index, item in enumerate(filter_names):
            if pab_filt in item:
                pab_idxs.append(index)
    

    idx_flags = np.zeros(len(integrated_sed_jy))
    idx_flags[ha_idxs[0]] = 1
    idx_flags[ha_idxs[1]] = 2
    idx_flags[ha_idxs[2]] = 1
    idx_flags[pab_idxs[0]] = 3
    idx_flags[pab_idxs[1]] = 4
    idx_flags[pab_idxs[2]] = 3
    integrated_spec_df = pd.DataFrame(zip(effective_waves_aa, integrated_sed_jy, idx_flags), columns=['wave_aa', 'integrated_spec_flux_jy', 'use_filter_flag'])
    integrated_spec_df.to_csv(f'/Users/brianlorenz/uncover/Data/integrated_specs/{id_msa}_integrated_spec.csv', index=False)
    def fint_pct(filts):
        red_row = sed_df[sed_df['filter'] == filts[0]]
        green_row = sed_df[sed_df['filter'] == filts[1]]
        blue_row = sed_df[sed_df['filter'] == filts[2]]
        cont_percentile = compute_cont_pct(blue_row.eff_wavelength.iloc[0], green_row.eff_wavelength.iloc[0], red_row.eff_wavelength.iloc[0], blue_row.flux.iloc[0], red_row.flux.iloc[0])
        return cont_percentile, red_row, green_row, blue_row
    ha_cont_pct, ha_red_row, ha_green_row, ha_blue_row = fint_pct(ha_filters)
    pab_cont_pct, pab_red_row, pab_green_row, pab_blue_row = fint_pct(pab_filters)
    ha_redflux = integrated_sed_jy[ha_idxs[0]]
    ha_greenflux = integrated_sed_jy[ha_idxs[1]]
    ha_blueflux = integrated_sed_jy[ha_idxs[2]]
    pab_redflux = integrated_sed_jy[pab_idxs[0]]
    pab_greenflux = integrated_sed_jy[pab_idxs[1]]
    pab_blueflux = integrated_sed_jy[pab_idxs[2]]
    ha_cont = np.percentile([ha_redflux, ha_blueflux], ha_cont_pct*100)
    pab_cont = np.percentile([pab_redflux, pab_blueflux], pab_cont_pct*100)
    
    # INtegrated spectrum
    ha_line_sed_intspec = (ha_green_flux_sed - ha_cont) /  line_transmissions[0]
    pab_line_sed_intspec = (pab_green_flux_sed - pab_cont) /  line_transmissions[1]
    line_ratio_from_sed_minus_intspec = compute_lineratio(ha_line_sed_intspec, pab_line_sed_intspec)

    # INtegrated spectrum with sed points
    ha_line = (ha_greenflux - ha_cont) / line_transmissions[0]
    pab_line = (pab_greenflux - pab_cont) / line_transmissions[1]
    line_ratio_from_spec = compute_lineratio(ha_line, pab_line)
    
    # From integrated spectrum, but with polynomial fit
    shifted_wave = spec_df['wave_aa'].to_numpy()
    f_jy = spec_df['flux'].to_numpy()
    line_waves = [6565*(1+redshift), 12821.7*(1+redshift)]
    line_masks = [np.logical_or(shifted_wave<(line_waves[i]-750), shifted_wave>(line_waves[i]+750)) for i in range(2)]
    wave_masks = [np.logical_and(shifted_wave>(line_waves[i]-3000), shifted_wave<(line_waves[i]+3000)) for i in range(2)]
    full_masks = [np.logical_and(line_masks[i], wave_masks[i]) for i in range(2)]
    ha_p5 = np.poly1d(np.polyfit(shifted_wave[full_masks[0]], f_jy[full_masks[0]], 5))
    pab_p5 = np.poly1d(np.polyfit(shifted_wave[full_masks[1]], f_jy[full_masks[1]], 5))
    ha_p5_erg = np.poly1d(np.polyfit(shifted_wave[full_masks[0]], f_lambda[full_masks[0]], 5))
    pab_p5_erg = np.poly1d(np.polyfit(shifted_wave[full_masks[1]], f_lambda[full_masks[1]], 5))
    # Recompute integrating polyfit
    ha_integrated_poly_abmag = observate.getSED(shifted_wave, ha_p5_erg(shifted_wave), filterlist=[filters[ha_idxs[1]]])
    pab_integrated_poly_abmag = observate.getSED(shifted_wave, pab_p5_erg(shifted_wave), filterlist=[filters[pab_idxs[1]]])
    ha_cont_fit = 10**(-0.4*(ha_integrated_poly_abmag-8.9))[0] # Currently integrating the continuum fit ove rthe line's filter
    pab_cont_fit = 10**(-0.4*(pab_integrated_poly_abmag-8.9))[0]

    ha_line_fit = (ha_greenflux - ha_cont_fit) / line_transmissions[0]
    pab_line_fit = (pab_greenflux - pab_cont_fit) / line_transmissions[1]
    line_ratio_from_spec_fit = compute_lineratio(ha_line_fit, pab_line_fit)

    # This one uses the actual SED points
    ha_line_fit_sed = (ha_green_flux_sed - ha_cont_fit) / line_transmissions[0]
    pab_line_fit_sed = (pab_green_flux_sed - pab_cont_fit) / line_transmissions[1]
    line_ratio_from_spec_fit_sed = compute_lineratio(ha_line_fit_sed, pab_line_fit_sed)

    # Here we use prospector spectrum and the actual SED points
    prospector_spec_df, prospector_sed_df = read_prospector(id_msa)
    shifted_wave_prospect = prospector_spec_df['wave_aa'].to_numpy()
    # If these lines are throwing an error, make sure to run compare_sed_spec_flux.py first
    f_jy_prospect = prospector_spec_df['spec_scaled_flux'].to_numpy()
    c_pros = 299792458 # m/s
    prospector_spec_df['flux_erg_aa'] = prospector_spec_df['spec_scaled_flux'] * (1e-23*1e10*c_pros / (prospector_spec_df['wave_aa']**2))
    f_lambda_prospect = prospector_spec_df['flux_erg_aa'].to_numpy()
    line_masks_prospect = [np.logical_or(shifted_wave_prospect<(line_waves[i]-750), shifted_wave_prospect>(line_waves[i]+750)) for i in range(2)]
    wave_masks_prospect = [np.logical_and(shifted_wave_prospect>(line_waves[i]-3000), shifted_wave_prospect<(line_waves[i]+3000)) for i in range(2)]
    full_masks_prospect = [np.logical_and(line_masks_prospect[i], wave_masks_prospect[i]) for i in range(2)]
    ha_p5_prospect = np.poly1d(np.polyfit(shifted_wave_prospect[full_masks_prospect[0]], f_jy_prospect[full_masks_prospect[0]], 5))
    pab_p5_prospect = np.poly1d(np.polyfit(shifted_wave_prospect[full_masks_prospect[1]], f_jy_prospect[full_masks_prospect[1]], 5))
    ha_p5_erg_prospect = np.poly1d(np.polyfit(shifted_wave_prospect[full_masks_prospect[0]], f_lambda_prospect[full_masks_prospect[0]], 5))
    pab_p5_erg_prospect = np.poly1d(np.polyfit(shifted_wave_prospect[full_masks_prospect[1]], f_lambda_prospect[full_masks_prospect[1]], 5))
    ha_integrated_poly_abmag_prospect = observate.getSED(shifted_wave_prospect, ha_p5_erg_prospect(shifted_wave_prospect), filterlist=[filters[ha_idxs[1]]])
    pab_integrated_poly_abmag_prospect = observate.getSED(shifted_wave_prospect, pab_p5_erg_prospect(shifted_wave_prospect), filterlist=[filters[pab_idxs[1]]])
    ha_cont_fit_prospect = 10**(-0.4*(ha_integrated_poly_abmag_prospect-8.9))[0] # Currently integrating the continuum fit ove rthe line's filter
    pab_cont_fit_prospect = 10**(-0.4*(pab_integrated_poly_abmag_prospect-8.9))[0]

    ha_line_fit_sed_prospect = (ha_green_flux_sed - ha_cont_fit_prospect) / line_transmissions[0]
    pab_line_fit_sed_prospect = (pab_green_flux_sed - pab_cont_fit_prospect) / line_transmissions[1]
    line_ratio_from_spec_fit_sed_prospect = compute_lineratio(ha_line_fit_sed_prospect, pab_line_fit_sed_prospect)

    ax_ha_main.plot(wavelength[full_masks[0]]/10000, ha_p5(wavelength[full_masks[0]]), ls='-', color='orange', marker='None')
    ax_pab_main.plot(wavelength[full_masks[1]]/10000, pab_p5(wavelength[full_masks[1]]), ls='-', color='orange', marker='None')
    ax_ha_main.plot(ha_green_row['eff_wavelength'], ha_cont_fit, color='orange', ls='None', marker='o')
    ax_pab_main.plot(pab_green_row['eff_wavelength'], pab_cont_fit, color='orange', ls='None', marker='o')
    ax_ha_main.plot(ha_green_row['eff_wavelength'], ha_cont_fit_prospect, color='magenta', ls='None', marker='o', zorder=1000)
    ax_pab_main.plot(pab_green_row['eff_wavelength'], pab_cont_fit_prospect, color='magenta', ls='None', marker='o', zorder=1000)
    ax_ha_main.plot(ha_green_row['eff_wavelength'], ha_greenflux, color='lime', ls='None', marker='o', alpha=0.5, zorder=100)
    ax_pab_main.plot(pab_green_row['eff_wavelength'], pab_greenflux, color='lime', ls='None', marker='o', alpha=0.5, zorder=100)

    n_figs = 4
    fontsize = 18
    fig2, axarr2 = plt.subplots(2, n_figs, figsize = (24, 10))
    # Top row ha:
    for i in range(n_figs):
        ax = axarr2[0, i]
        if i == 0:
            purple = 1
            trasm = 0
        else:
            purple = 0
            trasm = 0
        if i == 2:
            plt_prospect = 1
        else:
            plt_prospect = 0
        plot_sed_around_line(ax, ha_filters, sed_df, spec_df, redshift, 0, line_transmissions[0], ha_transmissions, id_msa, bootstrap=1000, plt_purple_merged_point=purple, plt_prospect=plt_prospect, show_trasm = trasm)
    for i in range(n_figs):
        if i == 0:
            purple = 1
            trasm = 0
        else:
            purple = 0
            trasm = 0
        if i == 2:
            plt_prospect = 1
        else:
            plt_prospect = 0
        ax = axarr2[1, i]
        plot_sed_around_line(ax, pab_filters, sed_df, spec_df, redshift, 1, line_transmissions[1], pab_transmissions, id_msa, bootstrap=1000, plt_purple_merged_point=purple, plt_prospect=plt_prospect, show_trasm=trasm)
    for j in range(2):
        # SED only method
        ax = axarr2[0,j]
        # Integrated Spectrum method
        ax = axarr2[0,j]
    axarr2[0,0].set_title('SED Only', fontsize=fontsize)
    axarr2[0,0].text(0.3, 1.1, f'Ratio: {round(sed_lineratio, 2)}', color='orange', transform=axarr2[0,0].transAxes, fontsize=fontsize)
    
    # Integrated spectrum - NOT REPSENTING PROPERLY
    # breakpoint()
    def get_filter_idxs(filts, spec_df):
        idx_list = []
        for filt in filts:
            blue_wave = filt.blue_edge / 10000
            red_wave = filt.red_edge / 10000
            idxs = np.logical_and(spec_df['wave'] > blue_wave, spec_df['wave'] < red_wave)
            idx_list.append(idxs)
        return idx_list
    idx_lists = get_filter_idxs([filters[ha_idxs[0]], filters[ha_idxs[2]], filters[pab_idxs[0]], filters[pab_idxs[2]]], spec_df)
    for idxs in idx_lists:
        axarr2[0,1].plot(spec_df['wave'][idxs], spec_df[idxs]['flux'], ls='-', marker='None', color='orange', lw=1)
        axarr2[1,1].plot(spec_df['wave'][idxs], spec_df[idxs]['flux'], ls='-', marker='None', color='orange', lw=1)
    def plot_int_spec(ha_rows, pab_rows, ha_int_fluxes, pab_int_fluxes, sed_greenfluxes, ax_ha, ax_pab):
        ha_red_row, ha_green_row, ha_blue_row = ha_rows
        pab_red_row, pab_green_row, pab_blue_row = pab_rows
        ha_redflux, ha_cont, ha_blueflux  = ha_int_fluxes
        pab_redflux, pab_cont, pab_blueflux  = pab_int_fluxes
        ha_green_flux_sed, pab_green_flux_sed = sed_greenfluxes
        ax_ha.plot(ha_green_row['eff_wavelength'], ha_cont, color='purple', ls='None', marker='o', zorder=100)
        ax_ha.plot(ha_red_row['eff_wavelength'], ha_redflux, color='orange', ls='None', marker='o', zorder=100)
        ax_ha.plot(ha_blue_row['eff_wavelength'], ha_blueflux, color='orange', ls='None', marker='o', zorder=100)
        ax_pab.plot(pab_red_row['eff_wavelength'], pab_redflux, color='orange', ls='None', marker='o', zorder=100)
        ax_pab.plot(pab_blue_row['eff_wavelength'], pab_blueflux, color='orange', ls='None', marker='o', zorder=100)
        ax_pab.plot(pab_green_row['eff_wavelength'], pab_cont, color='purple', ls='None', marker='o', zorder=100)
        ax_ha.plot([ha_red_row['eff_wavelength'].iloc[0], ha_blue_row['eff_wavelength'].iloc[0]], [ha_redflux, ha_blueflux],color='purple', ls='--', marker='None', zorder=100)
        ax_pab.plot([pab_red_row['eff_wavelength'].iloc[0], pab_blue_row['eff_wavelength'].iloc[0]], [pab_redflux, pab_blueflux],color='purple', ls='--', marker='None', zorder=100)
        ax_ha.plot([ha_green_row['eff_wavelength'].iloc[0], ha_green_row['eff_wavelength'].iloc[0]], [ha_green_flux_sed, ha_cont], color=connect_color, ls='-', marker='None', zorder=100)
        ax_pab.plot([pab_green_row['eff_wavelength'].iloc[0], pab_green_row['eff_wavelength'].iloc[0]], [pab_green_flux_sed, pab_cont], color=connect_color, ls='-', marker='None', zorder=100)
    ha_rows = ha_red_row, ha_green_row, ha_blue_row
    pab_rows = pab_red_row, pab_green_row, pab_blue_row 
    ha_int_fluxes = ha_redflux, ha_cont, ha_blueflux 
    pab_int_fluxes = pab_redflux, pab_cont, pab_blueflux 
    sed_greenfluxes = ha_green_flux_sed, pab_green_flux_sed
    plot_int_spec(ha_rows, pab_rows, ha_int_fluxes, pab_int_fluxes, sed_greenfluxes, axarr2[0,1], axarr2[1,1])
    axarr2[0,1].set_title('Integrated Spectrum', fontsize=fontsize)
    axarr2[0,1].text(0.3, 1.1, f'Ratio: {round(line_ratio_from_sed_minus_intspec, 2)}', color='magenta', transform=axarr2[0,1].transAxes, fontsize=fontsize)
    
    # Prospector
    axarr2[0,2].set_title('Prospector Spec Fit', fontsize=fontsize)
    axarr2[0,2].text(0.3, 1.1, f'Ratio: {round(line_ratio_from_spec_fit_sed_prospect, 2)}', color='purple', transform=axarr2[0,2].transAxes, fontsize=fontsize)
    axarr2[0,2].plot(wavelength[full_masks[0]]/10000, ha_p5_prospect(wavelength[full_masks[0]]), color='purple')
    axarr2[1,2].plot(wavelength[full_masks[1]]/10000, pab_p5_prospect(wavelength[full_masks[1]]), color='purple')
    axarr2[0,2].plot(ha_green_row['eff_wavelength'], ha_cont_fit_prospect, color='purple', ls='None', marker='o', zorder=100)
    axarr2[1,2].plot(pab_green_row['eff_wavelength'], pab_cont_fit_prospect, color='purple', ls='None', marker='o', zorder=100)
    axarr2[0,2].plot([ha_green_row['eff_wavelength'].iloc[0], ha_green_row['eff_wavelength'].iloc[0]], [ha_green_flux_sed, ha_cont_fit_prospect], color=connect_color, ls='-', marker='None', zorder=100)
    axarr2[1,2].plot([pab_green_row['eff_wavelength'].iloc[0], pab_green_row['eff_wavelength'].iloc[0]], [pab_green_flux_sed, pab_cont_fit_prospect], color=connect_color, ls='-', marker='None', zorder=100)
    
    # Int spec fit
    axarr2[0,3].set_title('Spec Fit, then integrate', fontsize=fontsize)
    axarr2[0,3].text(0.3, 1.1, f'Ratio: {round(line_ratio_from_spec_fit_sed, 2)}', color='dodgerblue', transform=axarr2[0,3].transAxes, fontsize=fontsize)
    axarr2[0,3].plot(wavelength[full_masks[0]]/10000, ha_p5(wavelength[full_masks[0]]), color='dodgerblue')
    axarr2[1,3].plot(wavelength[full_masks[1]]/10000, pab_p5(wavelength[full_masks[1]]), color='dodgerblue')
    axarr2[0,3].plot(ha_green_row['eff_wavelength'], ha_cont_fit, color='purple', ls='None', marker='o', zorder=100)
    axarr2[1,3].plot(pab_green_row['eff_wavelength'], pab_cont_fit, color='purple', ls='None', marker='o', zorder=100)
    axarr2[0,3].plot([ha_green_row['eff_wavelength'].iloc[0], ha_green_row['eff_wavelength'].iloc[0]], [ha_green_flux_sed, ha_cont_fit], color=connect_color, ls='-', marker='None', zorder=100)
    axarr2[1,3].plot([pab_green_row['eff_wavelength'].iloc[0], pab_green_row['eff_wavelength'].iloc[0]], [pab_green_flux_sed, pab_cont_fit], color=connect_color, ls='-', marker='None', zorder=100)

    # Emission fit truth
    fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
    line_ratio_from_emission = fit_df["ha_pab_ratio"].iloc[0]
    axarr2[0,0].text(-0.4, 1.21, f'Emission Fit Ratio: {round(line_ratio_from_emission, 2)}', color='black', transform=axarr2[0,0].transAxes, fontsize=fontsize)
    zqual_df = read_spec_cat()
    av_16, av_50, av_84 = read_catalog_av(id_msa, zqual_df)
    av_lineratio = compute_ratio_from_av(av_50)
    axarr2[0,3].text(0.72, 1.21, f'Prospector AV: {round((1/av_lineratio), 2)}', fontsize=fontsize, transform=axarr2[0,3].transAxes)

    fig2.savefig(f'/Users/brianlorenz/uncover/Figures/av_methods/{id_msa}_av_method.pdf')
    plt.close(fig2)


    fig, axarr = plt.subplots(2, 1, figsize=(6,8))
    ax_ha = axarr[0]
    ax_pab = axarr[1]
    
    ax_ha.plot(ha_red_row['eff_wavelength'], ha_redflux, color='red', ls='None', marker='o')
    ax_ha.plot(ha_green_row['eff_wavelength'], ha_greenflux, color='green', ls='None', marker='o', zorder=10)
    ax_ha.plot(ha_blue_row['eff_wavelength'], ha_blueflux, color='blue', ls='None', marker='o')
    ax_ha.plot(ha_green_row['eff_wavelength'], ha_cont, color='purple', ls='None', marker='o')
    ax_pab.plot(pab_red_row['eff_wavelength'], pab_redflux, color='red', ls='None', marker='o')
    ax_pab.plot(pab_green_row['eff_wavelength'], pab_greenflux, color='green', ls='None', marker='o', zorder=10)
    ax_pab.plot(pab_blue_row['eff_wavelength'], pab_blueflux, color='blue', ls='None', marker='o')
    ax_pab.plot(pab_green_row['eff_wavelength'], pab_cont, color='purple', ls='None', marker='o')
    for ax in axarr:
        ax.plot(spec_df['wave'], spec_df['flux'], ls='-', color='black', marker='None')
    ax_ha.set_xlim([ha_blue_row['eff_wavelength'].iloc[0]-0.1, ha_red_row['eff_wavelength'].iloc[0]+0.1])
    ax_pab.set_xlim([pab_blue_row['eff_wavelength'].iloc[0]-0.1, pab_red_row['eff_wavelength'].iloc[0]+0.1])
    ax_ha.text(0.1, 0.9, f'line ratio from int spec: {line_ratio_from_spec}', transform=ax_ha.transAxes)
    ax_ha.text(0.1, 0.8, f'int spec polyfit: {line_ratio_from_spec_fit}', transform=ax_ha.transAxes)
    ax_ha.text(0.1, 0.7, f'int spec polyfit sed: {line_ratio_from_spec_fit_sed}', transform=ax_ha.transAxes)
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/{id_msa}_lineratio.pdf')
    plt.close(fig)
    return line_ratio_from_spec, ha_line, pab_line, line_ratio_from_spec_fit, line_ratio_from_spec_fit_sed, line_ratio_from_spec_fit_sed_prospect

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

def compute_cont_pct(blue_wave, green_wave, red_wave, blue_flux, red_flux):
    total_wave_diff = blue_wave - red_wave
    line_wave_diff = green_wave - red_wave
    cont_percentile = line_wave_diff/total_wave_diff
    if red_flux>blue_flux:
        cont_percentile = 1-cont_percentile
    return cont_percentile

def get_cont_and_map(images, wht_images, pct, redshift, line_scaled_transmission):
    """Finds continuum as the percentile between the other two filters"""
    cont = np.percentile([images[0].data, images[2].data], pct*100, axis=0)
    err_cont = np.sqrt(((1-pct)*(1/np.sqrt(wht_images[0].data))))**2 + (pct*(1/np.sqrt(wht_images[2].data))**2)
    
    linemap = images[1].data - cont
    linemap_flux = linemap / (1+redshift)**2
    linemap_flux_scaled = linemap_flux / line_scaled_transmission
    
    err_linemap = np.sqrt(err_cont**2 + np.sqrt(1/wht_images[1].data)**2)
    err_linemap_flux = err_linemap / (1+redshift)**2
    err_linemap_flux_scaled = err_linemap_flux / line_scaled_transmission

    linemap_snr = linemap_flux_scaled/err_linemap_flux_scaled
    image = make_lupton_rgb(images[0].data, images[1].data, images[2].data, stretch=0.25)
    return cont, linemap_flux_scaled, image, linemap_snr

def compute_lineratio(ha_flux, pab_flux):
    # Updated calculation with 1/lambda**2
    lineratio = (ha_flux) / (pab_flux) / ((line_list[0][1] / line_list[1][1])**2)
    return lineratio

def flux_erg_to_jy(line_flux_erg, line_wave):
    c = 299792458 # m/s
    line_flux_jy = line_flux_erg / (1e-23*1e10*c / ((line_wave)**2))
    return line_flux_jy

# make_all_dustmap()
# make_dustmap(39744)
# make_dustmap(38163)
# make_dustmap(34114)

# make_dustmap(25147)
# make_dustmap(47875)


# make_dustmap(25774)
# make_dustmap(32111)


# make_3color(6291)
# make_3color(22755)
# make_3color(42203)
# make_3color(42213)