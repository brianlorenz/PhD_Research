from uncover_make_sed import read_sed
from make_dust_maps import make_3color, compute_cont_pct
from filter_integrals import get_transmission_at_line
from uncover_read_data import read_spec_cat, read_lineflux_cat, read_raw_spec
from uncover_sed_filters import unconver_read_filters
from sedpy import observate
from fit_emission_uncover import line_list
from astropy.io import ascii
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compare_spec_to_sed_3ways(id_msa):
    fig, axarr = plt.subplots(2, 3, figsize=(16,10))
    ax_noscale_ha = axarr[0,0]
    ax_scale_spec_ha = axarr[0,1]
    ax_scale_sed_ha = axarr[0,2]
    ax_noscale_pab = axarr[1,0]
    ax_scale_spec_pab = axarr[1,1]
    ax_scale_sed_pab = axarr[1,2]
    axes = [ax_noscale_ha, ax_scale_spec_ha, ax_scale_sed_ha, ax_noscale_pab, ax_scale_spec_pab, ax_scale_sed_pab]
    ha_axes = [ax_noscale_ha, ax_scale_spec_ha, ax_scale_sed_ha]
    pab_axes = [ax_noscale_pab, ax_scale_spec_pab, ax_scale_sed_pab]

    sed_df = read_sed(id_msa)
    spec_df = read_raw_spec(id_msa)

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

    ha_sedpy_filts = [ha_red_sedpy_filt, ha_sedpy_filt, ha_blue_sedpy_filt]
    pab_sedpy_filts = [pab_red_sedpy_filt, pab_sedpy_filt, pab_blue_sedpy_filt]

    ha_filters = ['f_'+filt for filt in ha_filters]
    pab_filters = ['f_'+filt for filt in pab_filters]

    # filt_dict, filters = unconver_read_filters()


    # wavelength = spec_df['wave_aa'].to_numpy()
    # f_lambda = spec_df['flux_erg_aa'].to_numpy()
    # sed_abmag = observate.getSED(wavelength, f_lambda, filterlist=filters)
    # sed_jy = 10**(-0.4*(sed_abmag-8.9))


    title_fontsize = 18
    fontsize = 14

    # Begin plotting
    use_ax = ax_noscale_ha
    use_ax.set_title('Ha no scaling', fontsize=title_fontsize)
    ha_waves = plot_sed_fluxes(use_ax, sed_df, ha_filters)
    plot_spec_fluxes(use_ax, spec_df, sed_df, ha_sedpy_filts, ha_filters)

    use_ax = ax_scale_spec_ha
    use_ax.set_title('Ha scale spec to SED', fontsize=title_fontsize)
    plot_sed_fluxes(use_ax, sed_df, ha_filters, spec_scaled=True)
    plot_spec_fluxes(use_ax, spec_df, sed_df, ha_sedpy_filts, ha_filters, scaled=True, line_name='Halpha')

    use_ax = ax_scale_sed_ha
    use_ax.set_title('Ha scale SED to spec', fontsize=title_fontsize)
    plot_sed_fluxes(use_ax, sed_df, ha_filters, scaled=True)
    plot_spec_fluxes(use_ax, spec_df, sed_df, ha_sedpy_filts, ha_filters)

    use_ax = ax_noscale_pab
    use_ax.set_title('PaB no scaling', fontsize=title_fontsize)
    pab_waves = plot_sed_fluxes(use_ax, sed_df, pab_filters)
    plot_spec_fluxes(use_ax, spec_df, sed_df, pab_sedpy_filts, pab_filters)

    use_ax = ax_scale_spec_pab
    use_ax.set_title('PaB scale spec to SED', fontsize=title_fontsize)
    plot_sed_fluxes(use_ax, sed_df, pab_filters, spec_scaled=True)
    plot_spec_fluxes(use_ax, spec_df, sed_df, pab_sedpy_filts, pab_filters, scaled=True, line_name='PaBeta')

    use_ax = ax_scale_sed_pab
    use_ax.set_title('PaB scale SED to spec', fontsize=title_fontsize)
    plot_sed_fluxes(use_ax, sed_df, pab_filters, scaled=True)
    plot_spec_fluxes(use_ax, spec_df, sed_df, pab_sedpy_filts, pab_filters)


    
    
    


    for ax in axes:
        ax.set_xlabel('Observed Wavelength', fontsize=fontsize)
        ax.set_ylabel('F_nu (Jy)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
    for ax in ha_axes:
        ax.set_xlim(ha_waves[2]-0.1, ha_waves[0]+0.1)
    for ax in pab_axes:
        ax.set_xlim(pab_waves[2]-0.1, pab_waves[0]+0.1)
    plt.tight_layout()

    fig.savefig(f'/Users/brianlorenz/uncover/Figures/simple_spec_compares/{id_msa}_simple_spec_comapre.pdf')
    plt.close('all')

def plot_sed_fluxes(ax, sed_df, filters, scaled=False, spec_scaled=False):
    for i in range(len(filters)):
        sed_row = sed_df[sed_df['filter'] == filters[i]]
        if i==0:
            red_wave = sed_row['eff_wavelength'].iloc[0]
            # red_flux = sed_row['integrated_spec_flux_jy'].iloc[0]
            red_flux = sed_row['flux'].iloc[0] # jy
            if scaled == True:
                red_flux = sed_row['spec_scaled_flux'].iloc[0] # jy
            # red_flux_erg_s_cm2 = compute_filter_F(red_flux, line_filts[0])
        if i == 1:
            green_wave = sed_row['eff_wavelength'].iloc[0]
            # green_flux = sed_row['integrated_spec_flux_jy'].iloc[0]
            green_flux = sed_row['flux'].iloc[0]
            int_spec_green_flux = sed_row['int_spec_flux'].iloc[0]
            # green_flux_erg_s_cm2 = compute_filter_F(green_flux, line_filts[1])
            if scaled == True:
                green_flux = sed_row['spec_scaled_flux'].iloc[0] # jy
        if i == 2:
            blue_wave = sed_row['eff_wavelength'].iloc[0]
            # blue_flux = sed_row['integrated_spec_flux_jy'].iloc[0]
            blue_flux = sed_row['flux'].iloc[0]
            # blue_flux_erg_s_cm2 = compute_filter_F(blue_flux, line_filts[2])
            if scaled == True:
                blue_flux = sed_row['spec_scaled_flux'].iloc[0] # jy

    ax.plot(red_wave, red_flux, color='red', ls='None', marker='o')
    ax.plot(green_wave, green_flux, color='green', ls='None', marker='o', zorder=10)
    ax.plot(blue_wave, blue_flux, color='blue', ls='None', marker='o')
    if spec_scaled == False:
        ax.plot(green_wave, int_spec_green_flux, color='lime', ls='None', marker='o', zorder=10)
    

    waves = [red_wave, green_wave, blue_wave]

    return waves

def plot_spec_fluxes(ax, spec_df, sed_df, sedpy_filters, filter_names, scaled=False, line_name='None'):
    plot_name = 'flux'
    
    red_sed_row = sed_df[sed_df['filter'] == filter_names[0]]
    blue_sed_row = sed_df[sed_df['filter'] == filter_names[2]]
    if scaled == True:
        blue_filt_edges = [sedpy_filters[2].blue_edge, sedpy_filters[2].red_edge]
        red_filt_edges = [sedpy_filters[0].blue_edge, sedpy_filters[0].red_edge]
        spec_range_blue = np.logical_and(spec_df['wave_aa'] > blue_filt_edges[0], spec_df['wave_aa'] < blue_filt_edges[1])
        spec_range_red = np.logical_and(spec_df['wave_aa'] > red_filt_edges[0], spec_df['wave_aa'] < red_filt_edges[1])
        
        spec_scale_blue = blue_sed_row['flux'] / np.median(spec_df[spec_range_blue]['flux'])
        spec_scale_red = red_sed_row['flux'] / np.median(spec_df[spec_range_red]['flux'])

        full_scale = np.median([spec_scale_blue.iloc[0], spec_scale_red.iloc[0]])

        if line_name == 'Halpha':
            f = open("scale_factors_ha.txt", "a")
        if line_name == 'PaBeta':
            f = open("scale_factors_pab.txt", "a")
        f.write(f"{full_scale}\n")
        f.close()

        print(spec_scale_blue)
        print(spec_scale_red)

        spec_df['scaled_flux'] = spec_df['flux'] * full_scale
        plot_name = 'scaled_flux'

        # filt_dict, filters = unconver_read_filters()
        # wavelength = spec_df['wave_aa'].to_numpy()
        # f_lambda = full_scale * spec_df['flux_erg_aa'].to_numpy()
        # sed_abmag = observate.getSED(wavelength, f_lambda, filterlist=filters)
        # sed_jy = 10**(-0.4*(sed_abmag-8.9))
        # sed_df['int_spec_flux_scaled'] = sed_jy
        green_sed_row = sed_df[sed_df['filter'] == filter_names[1]]
        green_wave = green_sed_row['eff_wavelength'].iloc[0]
        int_spec_green_flux_scaled = green_sed_row['int_spec_flux'].iloc[0] * full_scale
        
        ax.plot(green_wave, int_spec_green_flux_scaled, color='lime', ls='None', marker='o', zorder=10)
        
        
    ax.plot(spec_df['wave'], spec_df[plot_name], color='black', ls='-', marker='None')
    return


def compare_all_spec_to_sed_3ways(id_msa_list):
    for id_msa in id_msa_list:
        compare_spec_to_sed_3ways(id_msa)

if __name__ == "__main__":
    zqual_detected_df = ascii.read('/Users/brianlorenz/uncover/zqual_detected.csv').to_pandas()
    id_msa_list = zqual_detected_df['id_msa'].to_list()
    compare_all_spec_to_sed_3ways(id_msa_list)