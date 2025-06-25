from uncover_make_sed import read_sed
from simple_make_dustmap import make_3color, compute_cont_pct, compute_line
from filter_integrals import get_transmission_at_line
from uncover_read_data import read_spec_cat, read_lineflux_cat, get_id_msa_list, read_fluxcal_spec, read_supercat
from sedpy import observate
from fit_emission_uncover_wave_divide import line_list
from astropy.io import ascii
from simple_compute_lineratio import compute_lineratio
import numpy as np
import pandas as pd
from compute_av import get_nii_correction, get_fe_correction
from uncover_sed_filters import unconver_read_filters
import matplotlib.pyplot as plt
from plot_vals import *
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import math
from simple_abs_line_correction import fit_absorption_lines
from diagnostic_plots import plot_line_assessment



def calc_lineflux(id_msa, fluxcal_str=''):
    sed_df = read_sed(id_msa)
    mock_sed_df = ascii.read('/Users/brianlorenz/uncover/Data/integrated_specs/mock_ratio_15_flux_32_flat_shifted_47875.csv').to_pandas()
    sed_df = sed_df.join(mock_sed_df)
    zqual_df = read_spec_cat()
    redshift = zqual_df[zqual_df['id_msa']==id_msa]['z_spec'].iloc[0]

    paa_filters, paa_images, wht_paa_images, obj_segmap, paa_photfnus, paa_all_lines = make_3color(id_msa, line_index=2, plot=False, paalpha=True)
    paa_sedpy_name = paa_filters[1].replace('f', 'jwst_f')
    paa_sedpy_filt = observate.load_filters([paa_sedpy_name])[0]

    paa_red_sedpy_name = paa_filters[0].replace('f', 'jwst_f')
    paa_red_sedpy_filt = observate.load_filters([paa_red_sedpy_name])[0]
    paa_blue_sedpy_name = paa_filters[2].replace('f', 'jwst_f')
    paa_blue_sedpy_filt = observate.load_filters([paa_blue_sedpy_name])[0]

    paa_sedpy_filts = [paa_red_sedpy_filt, paa_sedpy_filt, paa_blue_sedpy_filt]

    paa_sedpy_width = paa_sedpy_filt.rectangular_width
    paa_sedpy_wave = paa_sedpy_filt.wave_effective
    paa_sedpy_transmission = get_transmission_at_line(paa_sedpy_filt, line_list[2][1] * (1+redshift), trasm_type='raw')

    paa_filters = ['f_'+filt for filt in paa_filters]

    paa_line_scaled_transmission = get_transmission_at_line(paa_sedpy_filt, line_list[2][1] * (1+redshift))

    paa_flux_erg_s_cm2, paa_cont_flux_erg_s_cm2_aa = measure_lineflux(id_msa, sed_df, redshift, paa_filters, paa_line_scaled_transmission, paa_sedpy_transmission, line_list[0][1], paa_sedpy_width, paa_sedpy_wave, paa_sedpy_filts)
    
    

    # Phot equivalent widths
    paa_phot_eq_width = paa_flux_erg_s_cm2 / paa_cont_flux_erg_s_cm2_aa

    # Integrated spectra
    spec_df = read_fluxcal_spec(id_msa)
    
    # For info about fluxcal factors
    paa_wave = np.logical_and(spec_df['rest_wave_aa'] > 6400, spec_df['rest_wave_aa'] < 6700) 
    paa_factor = spec_df['flux_calibrated_jy'][paa_wave].iloc[0] / spec_df['flux'][paa_wave].iloc[0]

    wavelength = spec_df['wave_aa'].to_numpy()
    f_lambda = spec_df['flux_calibrated_erg_aa'].to_numpy()
    filt_dict, filters = unconver_read_filters()
    filter_names = [sedpy_filt.name for sedpy_filt in filters]
    integrated_sed_abmag = observate.getSED(wavelength, f_lambda, filterlist=filters)
    integrated_sed_jy = 10**(-0.4*(integrated_sed_abmag-8.9))
    effective_waves_aa = sed_df['eff_wavelength']*10000
    int_spec_df = pd.DataFrame(zip(effective_waves_aa/10000, integrated_sed_jy), columns=['eff_wavelength', 'flux'])
    int_spec_df['filter'] = sed_df['filter']
    int_spec_df['err_flux'] = int_spec_df['flux']*0.01 # placehold for plot
    # plot_line_assessment([id_msa], int_spec_df=int_spec_df)
    int_spec_paa_flux_erg_s_cm2, _ = measure_lineflux(id_msa, int_spec_df, redshift, paa_filters, paa_line_scaled_transmission, paa_sedpy_transmission, line_list[0][1], paa_sedpy_width, paa_sedpy_wave, paa_sedpy_filts)
    
    lines_df = read_lineflux_cat()
    lines_df_row = lines_df[lines_df['id_msa'] == id_msa]
    paa_flux_cat_erg_s_cm2 = lines_df_row['f_PaA'].iloc[0] * 1e-20
    err_paa_flux_cat_erg_s_cm2 = lines_df_row['e_PaA'].iloc[0] * 1e-20

    fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting_paalpha_only/{id_msa}_emission_fits.csv').to_pandas()
    paa_flux_emission_fit = fit_df['flux'].iloc[0]
   

    paa_offset_factor = paa_flux_erg_s_cm2 / paa_flux_emission_fit

    return paa_flux_erg_s_cm2, paa_flux_cat_erg_s_cm2, err_paa_flux_cat_erg_s_cm2, paa_flux_emission_fit, int_spec_paa_flux_erg_s_cm2, paa_phot_eq_width

    # print(f'Ha/PaB: {(ha_flux/pab_flux) / (line_list[0][1]/line_list[1][1])**2}')
    # print(f'Cat Ha/PaB: {(ha_flux_cat_jy/pab_flux_cat_jy)}')

def measure_lineflux(id_msa, sed_df, redshift, filters, scaled_transmission, raw_transmission, line_wave, filter_width, filter_wave, line_filts):
    for i in range(len(filters)):
        sed_row = sed_df[sed_df['filter'] == filters[i]]
        if i==0:
            red_wave = sed_row['eff_wavelength'].iloc[0]
            # red_flux = sed_row['integrated_spec_flux_jy'].iloc[0]
            red_flux = sed_row['flux'].iloc[0] # jy
            red_flux_erg_s_cm2 = compute_filter_F(red_flux, line_filts[0])
        if i == 1:
            green_wave = sed_row['eff_wavelength'].iloc[0]
            # green_flux = sed_row['integrated_spec_flux_jy'].iloc[0]
            green_flux = sed_row['flux'].iloc[0]
            green_flux_erg_s_cm2 = compute_filter_F(green_flux, line_filts[1])
        if i == 2:
            blue_wave = sed_row['eff_wavelength'].iloc[0]
            # blue_flux = sed_row['integrated_spec_flux_jy'].iloc[0]
            blue_flux = sed_row['flux'].iloc[0]
            blue_flux_erg_s_cm2 = compute_filter_F(blue_flux, line_filts[2])
    def cor_he1(id_msa, blue_flux):
        he1_emfit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/helium/{id_msa}_emission_fits_helium1.csv').to_pandas()
        he1_flux = he1_emfit_df['flux'].iloc[1]
        he1_wave = 10830
        c = 299792458 # m/s
        he1_flux_fnu = he1_flux / ((c*1e10) / (he1_wave)**2) 
        he1_flux_jy = he1_flux_fnu / 1e-23
        he1_flux_filt_spread = he1_flux_jy / filter_width
        blue_flux = blue_flux - he1_flux_filt_spread  # THis is only about a 2% correction to total pab flux
        return blue_flux
    # blue_flux = cor_he1(id_msa, blue_flux)
    cont_percentile = compute_cont_pct(blue_wave, green_wave, red_wave, blue_flux, red_flux)
    # print(blue_flux)
    # print(green_flux)
    # print(red_flux)
    # cont_percentile2 = compute_cont_pct(blue_wave, green_wave, red_wave, blue_flux_erg_s_cm2, red_flux_erg_s_cm2)
    line_flux, cont_value, cont_flux_erg_s_cm2_aa = compute_line(cont_percentile, red_flux, green_flux, blue_flux, redshift, raw_transmission, filter_width, line_wave, calc_eq_width=True)
    if line_wave < 8000:
        line_name = 'ha'
    if line_wave > 8000:
        line_name = 'pab'
    # line_transmission = get_line_coverage(id_msa, line_filts[1], redshift=redshift, line_name=line_name)
    # line_flux = line_flux 
        
    return line_flux, cont_flux_erg_s_cm2_aa


def compute_line_already_erg(cont_pct, red_flx, green_flx, blue_flx, redshift, scaled_transmission, raw_transmission, filter_width, filter_wave, line_wave):
        cont_value = np.percentile([blue_flx, red_flx], cont_pct*100)

        line_value = green_flx - cont_value # erg/s/cm^2

        # Scale by raw transmission curve
        line_value = line_value / raw_transmission

        return line_value, cont_value




def calc_all_lineflux(id_msa_list, full_sample=False, fluxcal=True):
    if fluxcal:
        fluxcal_str = ''
    else:
        fluxcal_str = '_no_fluxcal'

    paa_sed_fluxes = []
    paa_cat_fluxes = []
    err_paa_cat_fluxes = []
    paa_emission_fits = []
    int_spec_paa_nocors = []
    paa_phot_eq_widths = []
    for id_msa in id_msa_list:
        paa_flux_erg_s_cm2, paa_flux_cat_erg_s_cm2, err_paa_flux_cat_erg_s_cm2, paa_flux_emission_fit, int_spec_paa_flux_erg_s_cm2, paa_phot_eq_width = calc_lineflux(id_msa, fluxcal_str=fluxcal_str)
        paa_sed_fluxes.append(paa_flux_erg_s_cm2)
        paa_cat_fluxes.append(paa_flux_cat_erg_s_cm2)
        err_paa_cat_fluxes.append(err_paa_flux_cat_erg_s_cm2)
        paa_emission_fits.append(paa_flux_emission_fit)
        int_spec_paa_nocors.append(int_spec_paa_flux_erg_s_cm2)
        paa_phot_eq_widths.append(paa_phot_eq_width)
    emission_offset_df = pd.DataFrame(zip(id_msa_list, paa_sed_fluxes, paa_cat_fluxes, err_paa_cat_fluxes, paa_emission_fits, int_spec_paa_nocors, paa_phot_eq_widths), columns=['id_msa', 'paa_sed_flux', 'paa_cat_flux', 'err_paa_cat_flux', 'paa_emfit_flux',  'int_spec_paa_nocor', 'paa_phot_eq_width'])
    emission_offset_df['paa_sed_div_cat'] = emission_offset_df['paa_sed_flux'] / emission_offset_df['paa_cat_flux']
    emission_offset_df['paa_sed_div_emfit'] = emission_offset_df['paa_sed_flux'] / emission_offset_df['paa_emfit_flux']
    emission_offset_df['paa_cat_div_emfit'] = emission_offset_df['paa_cat_flux'] / emission_offset_df['paa_emfit_flux']
    
    
    emission_offset_df.to_csv(f'/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df_paalpha_only.csv', index=False)
    print(f'\n\n\n')
    print(f'median offset in Ha {np.median(emission_offset_df["paa_sed_div_emfit"])}')
    
def compute_filter_F(f_nu_jy, sedpy_filt):
    filter_wave = sedpy_filt.wave_effective
    filter_width = sedpy_filt.rectangular_width
    median_filter_trasm = np.median(sedpy_filt.transmission)
    # filter_area = median_filter_trasm * filter_width

    f_nu = f_nu_jy * 1e-23

    c = 299792458 # m/s
    f_lambda_aa = f_nu * ((c*1e10) / (filter_wave)**2) 
    f_total = f_lambda_aa * filter_width  # erg/s/cm^2
    return f_total

def plot_sed_vs_intspec():
    lineflux_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df.csv').to_pandas()
    fig, axarr = plt.subplots(1,2, figsize=(12,6))
    ax_ha = axarr[0]
    ax_pab = axarr[1]
    ax_ha.plot(lineflux_df['int_spec_ha_nocor'], lineflux_df['ha_sed_flux'], marker='o', color='black', ls='None')
    ax_pab.plot(lineflux_df['int_spec_pab_fecor'], lineflux_df['pab_sed_flux'], marker='o', color='black', ls='None')
    
    ax_ha.set_ylabel('SED Ha flux', fontsize=14)
    ax_ha.set_xlabel('Integrated Spectrum Ha flux', fontsize=14)
    ax_pab.set_ylabel('SED PaB flux', fontsize=14)
    ax_pab.set_xlabel('Integrated Spectrum PaB flux', fontsize=14)

    line_p1 = np.array([-20, -20])
    line_p2 = np.array([-15, -15])
    def get_distance(datapoint):
        distance = np.cross(line_p2-line_p1,datapoint-line_p1)/np.linalg.norm(line_p2-line_p1)
        return distance
    ha_distances = []
    pab_distances = []
    for i in range(len(lineflux_df)):
        log_ha_datapoint = (np.log10(lineflux_df['int_spec_ha_nocor'].iloc[i]), np.log10(lineflux_df['ha_sed_flux'].iloc[i]))
        log_pab_datapoint = (np.log10(lineflux_df['int_spec_pab_fecor'].iloc[i]), np.log10(lineflux_df['pab_sed_flux'].iloc[i]))
        ha_distances.append(get_distance(np.array(log_ha_datapoint)))
        pab_dist = get_distance(np.array(log_pab_datapoint))
        if pd.isnull(pab_dist):
            continue
        pab_distances.append(pab_dist)
    ha_distances = np.abs(ha_distances)
    pab_distances = np.abs(pab_distances)
    median_ha_offset = np.median(ha_distances)
    scatter_ha_offset = np.std(ha_distances)
    median_pab_offset = np.median(pab_distances)
    scatter_pab_offset = np.std(pab_distances)

    start_scatter_text_x = 0.02
    start_scatter_text_y = 0.94
    scatter_text_sep = 0.07
    ax_ha.text(start_scatter_text_x, start_scatter_text_y, f'Offset: {median_ha_offset:0.2f}', transform=ax_ha.transAxes, fontsize=12)
    ax_ha.text(start_scatter_text_x, start_scatter_text_y-scatter_text_sep, f'Scatter: {scatter_ha_offset:0.2f}', transform=ax_ha.transAxes, fontsize=12)
    ax_pab.text(start_scatter_text_x, start_scatter_text_y, f'Offset: {median_pab_offset:0.2f}', transform=ax_pab.transAxes, fontsize=12)
    ax_pab.text(start_scatter_text_x, start_scatter_text_y-scatter_text_sep, f'Scatter: {scatter_pab_offset:0.2f}', transform=ax_pab.transAxes, fontsize=12)
    

    for ax in axarr:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=14)
        ax.plot([1e-20, 1e-14], [1e-20, 1e-14], ls='--', color='red', marker='None')
        ax.set_xlim([5e-19, 1e-15])
        ax.set_ylim([5e-19, 1e-15])
    fig.savefig('/Users/brianlorenz/uncover/Figures/paper_figures/sed_vs_emfit_zcompare_intspec.pdf', bbox_inches='tight')

def plot_spec_vs_intspec(color='', full_sample=False):
    lineflux_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df.csv').to_pandas()
    full_lineratio_data_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineratio_av_df.csv').to_pandas()
    supercat_df = read_supercat()
    id_msa_list = get_id_msa_list()
    lineratio_data_df = full_lineratio_data_df[full_lineratio_data_df['id_msa'].isin(id_msa_list)]
    save_str = ''
    if full_sample == True:
        lineflux_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df_all.csv').to_pandas()
        save_str = '_all'

    fig, axarr = plt.subplots(1,3, figsize=(21.5,6))
    ax_ha = axarr[0]
    ax_pab = axarr[1]
    ax_av = axarr[2]
    ax_list = [ax_ha, ax_pab]
    
    
    ax_ha.set_xlabel('Emission Fit Ha+NII flux', fontsize=14)
    ax_ha.set_ylabel('Integrated Spectrum Ha+NII flux', fontsize=14)
    ax_pab.set_xlabel('Emission Fit PaB+FeII flux', fontsize=14)
    ax_pab.set_ylabel('Integrated Spectrum PaB+FeII flux', fontsize=14)
    ax_av.set_xlabel('Emission Fit lineratio', fontsize=14)
    ax_av.set_ylabel('Integrated Spectrum lineratio', fontsize=14)

    cmap = mpl.cm.inferno
            
    line_p1 = np.array([-20, -20])
    line_p2 = np.array([-15, -15])
    def get_distance(datapoint):
        distance = np.cross(line_p2-line_p1,datapoint-line_p1)/np.linalg.norm(line_p2-line_p1)
        return distance
    ha_distances = []
    pab_distances = []
    for i in range(len(lineflux_df)):
        id_msa = lineflux_df['id_msa'].iloc[i]
        lineratio_data_row = lineratio_data_df[lineratio_data_df['id_msa'] == id_msa]
        fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
        fit_ha_flux = fit_df['flux'].iloc[0]
        ha_snr = fit_df['signal_noise_ratio'].iloc[0]
        pab_snr = fit_df['signal_noise_ratio'].iloc[1]
        ha_eqw_fit = fit_df.iloc[0]['equivalent_width_aa']
        pab_eqw_fit = fit_df.iloc[1]['equivalent_width_aa']
        fe_cor_df_indiv = ascii.read('/Users/brianlorenz/uncover/Data/generated_tables/fe_cor_df_indiv.csv').to_pandas()
        fe_cor_df_row = fe_cor_df_indiv[fe_cor_df_indiv['id_msa'] == id_msa]
        if len(fe_cor_df_row) == 0:
            fit_pab_flux = fit_df['flux'].iloc[1]
            only_pab=True
        else:
            fe_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting_fe_only/{id_msa}_emission_fits.csv').to_pandas()
            fe_flux = fe_df['flux'].iloc[0]
            fit_pab_flux = fit_df['flux'].iloc[1] + fe_flux
            only_pab=False

        if len(lineratio_data_row) == 0:
            fit_lineratio = -99
            int_spec_lineratio = -99
        else:
            fit_lineratio = 1/lineratio_data_row['emission_fit_lineratio'].iloc[0]
            # int_spec_lineratio = lineflux_df['int_spec_pab_fecor'].iloc[i] / lineflux_df['int_spec_ha_niicor'].iloc[i]
            supercat_row = supercat_df[supercat_df['id_msa']==id_msa]
            id_dr3 = supercat_row['id'].iloc[0]
            ha_absorp_eqw_fit, pab_absorp_eqw_fit = fit_absorption_lines(id_dr3)
            pab_cor_flux = lineflux_df['int_spec_pab_fecor'].iloc[i]
            int_spec_lineratio = 1/compute_lineratio(lineflux_df['int_spec_ha_niicor'].iloc[i], pab_cor_flux, ha_eqw_fit, pab_eqw_fit, ha_absorp_eqw_fit, pab_absorp_eqw_fit)
        
        if color == 'pab_snr':
            norm = mpl.colors.LogNorm(vmin=1, vmax=60) 
            rgba_ha = cmap(norm(pab_snr))
            rgba_pab = cmap(norm(pab_snr))
            rgba_av = cmap(norm(pab_snr))
            label='PaB SNR'
        if color == 'line_snr':
            norm = mpl.colors.LogNorm(vmin=1, vmax=100) 
            # print(ha_snr)
            rgba_ha = cmap(norm(ha_snr))
            rgba_pab = cmap(norm(pab_snr))
            print(pab_snr)
            print(' ')
            rgba_av = 'black'
            label='Line SNR'
        if color == 'fe_or_not':
            rgba_ha = 'black'
            rgba_pab='black'
            rgba_av='black'
            if only_pab == True:
                rgba_pab='red'
                rgba_av='red'

        # if ha_snr > 30:
        #     continue
        
        ax_ha.plot(fit_ha_flux, lineflux_df['int_spec_ha_nocor'].iloc[i], marker='o', color=rgba_ha, ls='None', mec='black')
        ax_pab.plot(fit_pab_flux, pab_cor_flux, marker='o', color=rgba_pab, ls='None', mec='black')
        ax_av.plot(fit_lineratio, int_spec_lineratio, marker='o', color=rgba_av, ls='None', mec='black')
    
        # print([fit_lineratio, int_spec_lineratio])
        # if lineflux_df['int_spec_pab_nocor'].iloc[i] > 1e-25:
        if int_spec_lineratio > 0.1:
            ax_ha.text(fit_ha_flux, lineflux_df['int_spec_ha_nocor'].iloc[i], f'{id_msa}')
            ax_pab.text(fit_pab_flux, lineflux_df['int_spec_pab_fecor'].iloc[i], f'{id_msa}')
            ax_av.text(fit_lineratio, int_spec_lineratio, f'{id_msa}')

        log_ha_datapoint = (np.log10(fit_ha_flux), np.log10(lineflux_df['int_spec_ha_nocor'].iloc[i]))
        log_pab_datapoint = (np.log10(fit_pab_flux), np.log10(lineflux_df['int_spec_pab_fecor'].iloc[i]))
        log_av_datapoint = (np.log10(fit_lineratio), np.log10(int_spec_lineratio))

        ha_distances.append(get_distance(np.array(log_ha_datapoint)))
        pab_dist = get_distance(np.array(log_pab_datapoint))
        av_dist = get_distance(np.array(log_av_datapoint))
        pab_distances.append(pab_dist)
    ha_distances = np.abs(ha_distances)
    pab_distances = np.abs(pab_distances)
    ha_distances = [x for x in ha_distances if not math.isnan(x)]
    median_ha_offset = np.median(ha_distances)
    scatter_ha_offset = np.std(ha_distances)
    median_pab_offset = np.median(pab_distances)
    scatter_pab_offset = np.std(pab_distances)

    start_scatter_text_x = 0.02
    start_scatter_text_y = 0.94
    scatter_text_sep = 0.07
    ax_ha.text(start_scatter_text_x, start_scatter_text_y, f'Offset: {median_ha_offset:0.2f}', transform=ax_ha.transAxes, fontsize=12)
    ax_ha.text(start_scatter_text_x, start_scatter_text_y-scatter_text_sep, f'Scatter: {scatter_ha_offset:0.2f}', transform=ax_ha.transAxes, fontsize=12)
    ax_pab.text(start_scatter_text_x, start_scatter_text_y, f'Offset: {median_pab_offset:0.2f}', transform=ax_pab.transAxes, fontsize=12)
    ax_pab.text(start_scatter_text_x, start_scatter_text_y-scatter_text_sep, f'Scatter: {scatter_pab_offset:0.2f}', transform=ax_pab.transAxes, fontsize=12)
    
    if color != 'fe_or_not':
        sm =  ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax_av)
        cbar.set_label(label, fontsize=14)
        cbar.ax.tick_params(labelsize=14)
    else:
        line_cor = Line2D([0], [0], color='black', marker='o', markersize=6, ls='None', mec='black')
        line_nocor = Line2D([0], [0], color='red', marker='o', markersize=6, ls='None', mec='black')
        custom_lines = [line_cor, line_nocor]
        custom_labels = ['Fe detected in spec', 'Fe not in spec']
        ax_pab.legend(custom_lines, custom_labels, loc=4, fontsize=12)
    
    for ax in axarr:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=14)
    for ax in ax_list:
        ax.plot([1e-20, 1e-14], [1e-20, 1e-14], ls='--', color='red', marker='None')
        ax.set_xlim([5e-19, 1e-15])
        ax.set_ylim([5e-19, 1e-15])
    ax_av.plot([1/50, 1], [1/50, 1], ls='--', color='red', marker='None')
    ax_av.set_xlim([1/40, 1/1.5])
    ax_av.set_ylim([1/40, 1/1.5])
    y_tick_locs = [0.03, 0.055, 1/10, 1/5, 1/2]
    y_tick_labs = ['0.03', '0.055', '0.1', '0.2', '0.5']
    ax_av.set_yticks(y_tick_locs)
    ax_av.set_yticklabels(y_tick_labs)
    ax_av.set_xticks(y_tick_locs)
    ax_av.set_xticklabels(y_tick_labs)
    for ax in axarr:
        scale_aspect(ax)

    fig.savefig(f'/Users/brianlorenz/uncover/Figures/paper_figures/sed_vs_emfit_zcompare_intspec3_{color}{save_str}.pdf', bbox_inches='tight')



if __name__ == "__main__":
    # calc_lineflux(47875)
    # calc_lineflux(14573)
    # calc_lineflux(25774)
    # calc_lineflux(39855)
    
    # id_msa_list = get_id_msa_list(full_sample=False, referee_sample=True)
    paa_sample = [15350, 17089, 18045, 19283, 25774, 27621, 29398, 33157, 38987, 42203, 42238, 43497, 48463]
    supercat_df = read_supercat()
    id_dr3s = []
    for id_msa in paa_sample:
        id_dr3 = supercat_df[supercat_df['id_msa'] == id_msa]['id'].iloc[0]
        id_dr3s.append(id_dr3)
    breakpoint()
    calc_all_lineflux(paa_sample, fluxcal=True)  

    # plot_sed_vs_intspec()
    # plot_spec_vs_intspec('pab_snr')
    # plot_spec_vs_intspec('line_snr', full_sample=False)
    # plot_spec_vs_intspec('fe_or_not')
    import sys
    sys.exit()

    id_msa_list = get_id_msa_list(full_sample=False)
    calc_all_lineflux(id_msa_list, full_sample=False, fluxcal=True)  
    pass 