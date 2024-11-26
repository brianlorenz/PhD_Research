import matplotlib.pyplot as plt
from astropy.io import ascii
import numpy as np
from compute_av import ha_factor, pab_factor, compute_ratio_from_av, compute_ha_pab_av, compute_ha_pab_av_from_dustmap, read_catalog_av
from uncover_read_data import read_supercat, read_raw_spec, read_spec_cat, read_segmap, read_SPS_cat
import pandas as pd
import matplotlib as mpl
from make_dust_maps import make_3color
from sedpy import observate

add_str = ''
colorbar = True
cbar_min = 1.3
cbar_max = 2.4
cbar_label = 'redshift'

def generate_filtered_lineratio_df(aper_size="None"):
    # Read in the data
    if aper_size != "None":
        lineratio_df = ascii.read(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/lineratio_df{add_str}_aper{aper_size}.csv', data_start=1).to_pandas()
    else:
        lineratio_df = ascii.read(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/lineratio_df{add_str}.csv', data_start=1).to_pandas()
    zqual_df_cont_covered = ascii.read('/Users/brianlorenz/uncover/zqual_df_cont_covered.csv').to_pandas()
    # supercat_df = read_supercat()
    good_rows = np.logical_and(zqual_df_cont_covered['ha_trasm_flag'] == 0, zqual_df_cont_covered['pab_trasm_flag'] == 0)
    good_ids = zqual_df_cont_covered[good_rows]['id_msa']
    filtered_linartio_df = lineratio_df[lineratio_df['id_msa'].isin(good_ids)]

    # Read in prospector fits
    zqual_df = read_spec_cat()
    av_16s = []
    av_50s = []
    av_84s = []
    for i in range(len(filtered_linartio_df)):
        id_msa = filtered_linartio_df.iloc[i]['id_msa']
        av_16, av_50, av_84 = read_catalog_av(id_msa, zqual_df)
        av_16s.append(av_16)
        av_50s.append(av_50)
        av_84s.append(av_84)             
    filtered_linartio_df['av_16'] = av_16s
    filtered_linartio_df['av_50'] = av_50s
    filtered_linartio_df['av_84'] = av_84s


    # Convert the line ratios to av
    sed_av = compute_ha_pab_av(1/filtered_linartio_df['sed_lineratio'])
    sed_av_16 = compute_ha_pab_av(1/filtered_linartio_df['sed_lineratio_16'])
    sed_av_84 = compute_ha_pab_av(1/filtered_linartio_df['sed_lineratio_84'])
    emission_av = compute_ha_pab_av(1/filtered_linartio_df['emission_fit_lineratio'])
    emission_av_high = compute_ha_pab_av(1/(filtered_linartio_df['emission_fit_lineratio']-filtered_linartio_df['err_emission_fit_lineratio_low']))
    emission_av_low = compute_ha_pab_av(1/(filtered_linartio_df['err_emission_fit_lineratio_high']+filtered_linartio_df['emission_fit_lineratio']))

    filtered_linartio_df['sed_av'] = sed_av
    filtered_linartio_df['sed_av_16'] = sed_av_16
    filtered_linartio_df['sed_av_84'] = sed_av_84
    filtered_linartio_df['emission_fit_av'] = emission_av
    filtered_linartio_df['emission_fit_av_low'] = emission_av_low
    filtered_linartio_df['emission_fit_av_high'] = emission_av_high

    if aper_size != "None":
        filtered_linartio_df.to_csv(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/filtered_lineratio_df{add_str}_aper{aper_size}.csv', index=False)
    else:
        filtered_linartio_df.to_csv(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/filtered_lineratio_df{add_str}.csv', index=False)
    return 

def make_av_compare_figure(regenerate = False):
    if regenerate == True:
        generate_filtered_lineratio_df()

    filtered_lineratio_df = ascii.read(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/filtered_lineratio_df{add_str}.csv').to_pandas()
    zqual_df_cont_covered = ascii.read('/Users/brianlorenz/uncover/zqual_df_cont_covered.csv').to_pandas()

    # Setup plots
    fig, axarr = plt.subplots(1, 2, figsize=(12,6))
    ax_prospector = axarr[0]
    ax_sed = axarr[1]

    y_var = 'av_50'

    for i in range(len(filtered_lineratio_df)):

        row = filtered_lineratio_df.iloc[i]
        id_msa = int(row['id_msa'])
        zqual_row = zqual_df_cont_covered[zqual_df_cont_covered['id_msa'] == id_msa]
        fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
        ha_sigma = fit_df['sigma'].iloc[0]
        pab_sigma = fit_df['sigma'].iloc[1]

        pab_filters, pab_images, wht_pab_images, obj_segmap = make_3color(id_msa, line_index=1, plot=False)
        pab_red_sedpy_name = pab_filters[0].replace('f', 'jwst_f')
        pab_red_sedpy_filt = observate.load_filters([pab_red_sedpy_name])[0]
        pab_red_wave_obs = pab_red_sedpy_filt.wave_effective
        z_spec = zqual_row['z_spec'].iloc[0]
        pab_red_wave_rest = pab_red_wave_obs / (1+z_spec)
        print(pab_red_wave_rest)

        cmap = mpl.cm.inferno
        norm = mpl.colors.Normalize(vmin=cbar_min, vmax=cbar_max) 
        rgba = cmap(norm(z_spec))
        if colorbar == False:
            rgba = 'black'

        prospector_av_err=np.array([[row[y_var]-row[y_var.replace('50','16')], row[y_var.replace('50','84')]-row[y_var]]]).T
        emission_av_err=np.array([[row['emission_fit_av']-row['emission_fit_av_low'], row['emission_fit_av_high']-row['emission_fit_av']]]).T

        ax_prospector.errorbar(row['emission_fit_av'], row['av_50'], xerr=emission_av_err, yerr=prospector_av_err, ls='None', marker='o', color=rgba, mec='black')

        sed_av_err=np.array([[row['sed_av']-row['sed_av_16'], row['sed_av_84']-row['sed_av']]]).T
        ax_sed.errorbar(row['emission_fit_av'], row['sed_av'], xerr=emission_av_err, yerr=sed_av_err, ls='None', marker='o', color=rgba, mec='black')
        ax_sed.text(row['emission_fit_av'], row['sed_av'], f'{id_msa}')

    if colorbar == True:
        sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax_sed)
        cbar.set_label(cbar_label, fontsize=16)
        cbar.ax.tick_params(labelsize=16)

    for ax in axarr:
        ax.set_xlabel('Emission Fit A$_V$')
        ax.tick_params(labelsize=12)
        ax.plot([-100, 100], [-100, 100], ls='--', color='red', marker='None')
        ax.set_xlim(-1,2)

    ax_prospector.set_ylim(-1,2)
    ax_prospector.set_ylabel('Prospector Fit A$_V$')
    
    ax_sed.set_ylim(-1,3.5)
    ax_sed.set_ylabel('SED A$_V$')

    fig.savefig(f'/Users/brianlorenz/uncover/Figures/paper_figures/av_comparison{add_str}.pdf')
    
    
    return

def intspec_sed_compare():
    fig, axarr = plt.subplots(1, 3, figsize=(18,6))
    ax_ha = axarr[0]
    ax_pab = axarr[1]
    ax_lineratio = axarr[2]

    lineratio_df = ascii.read(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/filtered_lineratio_df{add_str}.csv').to_pandas()
    zqual_df_cont_covered = ascii.read('/Users/brianlorenz/uncover/zqual_df_cont_covered.csv').to_pandas()

    for i in range(len(lineratio_df)):
        row = lineratio_df.iloc[i]
        id_msa = int(row['id_msa'])

        zqual_row = zqual_df_cont_covered[zqual_df_cont_covered['id_msa'] == id_msa]

        pab_filters, pab_images, wht_pab_images, obj_segmap = make_3color(id_msa, line_index=1, plot=False)
        pab_red_sedpy_name = pab_filters[0].replace('f', 'jwst_f')
        pab_red_sedpy_filt = observate.load_filters([pab_red_sedpy_name])[0]
        pab_red_wave_obs = pab_red_sedpy_filt.wave_effective
        z_spec = zqual_row['z_spec'].iloc[0]
        pab_red_wave_rest = pab_red_wave_obs / (1+z_spec)
        print(pab_red_wave_rest)
        fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
        ha_sigma = fit_df['sigma'].iloc[0]
        pab_sigma = fit_df['sigma'].iloc[1]

        cmap = mpl.cm.inferno
        norm = mpl.colors.Normalize(vmin=cbar_min, vmax=cbar_max) 
        rgba = cmap(norm(z_spec))
        
        if colorbar == False:
            rgba = 'black'

        sed_av_err=np.array([[row['sed_av']-row['sed_av_16'], row['sed_av_84']-row['sed_av']]]).T
        ha_sed_err = np.array([[row['sed_ha_compare']-row['sed_ha_compare_16'], row['sed_ha_compare_84']-row['sed_ha_compare']]]).T
        pab_sed_err = np.array([[row['sed_pab_compare']-row['sed_pab_compare_16'], row['sed_pab_compare_84']-row['sed_pab_compare']]]).T
        ax_ha.errorbar(row['int_spec_ha_compare'], row['sed_ha_compare'], yerr=ha_sed_err, ls='None', marker='o', color=rgba, mec='black')
        ax_pab.errorbar(row['int_spec_pab_compare'], row['sed_pab_compare'], yerr=pab_sed_err, ls='None', marker='o', color=rgba, mec='black')
        ax_lineratio.errorbar(row['integrated_spec_lineratio'], row['sed_lineratio'], yerr=sed_av_err, ls='None', marker='o', color=rgba, mec='black')

    if colorbar == True:
        sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax_lineratio)
        cbar.set_label(cbar_label, fontsize=16)
        cbar.ax.tick_params(labelsize=16)

    ax_ha.set_title('Halpha')
    ax_pab.set_title('PaBeta')
    ax_lineratio.set_title('Line Ratio')
    for ax in [ax_ha, ax_pab]:
        ax.set_xlabel('Integrated Spectrum Flux')
        ax.set_ylabel('SED Flux')
        ax.set_xscale('log')
        ax.set_yscale('log')
    for ax in axarr:
        ax.tick_params(labelsize=12)
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        ax.plot([-100, 100], [-100, 100], ls='--', color='red', marker='None')
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

    ax_lineratio.set_xlabel('Integrated Spectrum Line Ratio')
    ax_lineratio.set_ylabel('SED Line Ratio')

    # ax_lineratio.set_xlim(0.8*np.min(lineratio_df['integrated_spec_lineratio']), 1.2*np.max(lineratio_df['integrated_spec_lineratio']))
    # ax_lineratio.set_ylim(0.8*np.min(lineratio_df['sed_lineratio']), 1.2*np.max(lineratio_df['sed_lineratio']))

    # ax_ha.set_xlim(0.8*np.min(lineratio_df['int_spec_ha_compare']), 1.2*np.max(lineratio_df['int_spec_ha_compare']))
    # ax_pab.set_xlim(0.8*np.min(lineratio_df['int_spec_pab_compare']), 1.2*np.max(lineratio_df['int_spec_pab_compare']))
    # ax_ha.set_ylim(0.8*np.min(lineratio_df['sed_ha_compare']), 1.2*np.max(lineratio_df['sed_ha_compare']))
    # ax_pab.set_ylim(0.8*np.min(lineratio_df['sed_pab_compare']), 1.2*np.max(lineratio_df['sed_pab_compare']))

    


    fig.savefig(f'/Users/brianlorenz/uncover/Figures/paper_figures/intspec_sed_compare{add_str}.pdf')

def emission_sed_compare():
    fig, axarr = plt.subplots(1, 3, figsize=(18,6))
    ax_ha = axarr[0]
    ax_pab = axarr[1]
    ax_lineratio = axarr[2]

    lineratio_df = ascii.read(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/filtered_lineratio_df{add_str}.csv').to_pandas()
    zqual_df_cont_covered = ascii.read('/Users/brianlorenz/uncover/zqual_df_cont_covered.csv').to_pandas()

    for i in range(len(lineratio_df)):
        row = lineratio_df.iloc[i]
        id_msa = int(row['id_msa'])

        zqual_row = zqual_df_cont_covered[zqual_df_cont_covered['id_msa'] == id_msa]

        pab_filters, pab_images, wht_pab_images, obj_segmap = make_3color(id_msa, line_index=1, plot=False)
        pab_red_sedpy_name = pab_filters[0].replace('f', 'jwst_f')
        pab_red_sedpy_filt = observate.load_filters([pab_red_sedpy_name])[0]
        pab_red_wave_obs = pab_red_sedpy_filt.wave_effective
        z_spec = zqual_row['z_spec'].iloc[0]
        pab_red_wave_rest = pab_red_wave_obs / (1+z_spec)
        print(pab_red_wave_rest)
        fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
        ha_sigma = fit_df['sigma'].iloc[0]
        pab_sigma = fit_df['sigma'].iloc[1]
        ha_flux = fit_df['flux'].iloc[0]
        ha_wave = fit_df['line_center_rest'].iloc[0]
        pab_flux = fit_df['flux'].iloc[1]
        pab_wave = fit_df['line_center_rest'].iloc[1]
        c = 299792458 # m/s
        ha_flux_jy = ha_flux / (1e-23*1e10*c / ((ha_wave)**2))
        pab_flux_jy = pab_flux / (1e-23*1e10*c / ((pab_wave)**2))
        spec_scale_factor = row['spec_scale_factor']
        ha_flux_jy = ha_flux_jy * spec_scale_factor
        pab_flux_jy = pab_flux_jy * spec_scale_factor

        cmap = mpl.cm.inferno
        norm = mpl.colors.Normalize(vmin=cbar_min, vmax=cbar_max) 
        rgba = cmap(norm(z_spec))
        
        if colorbar == False:
            rgba = 'black'

        sed_av_err=np.array([[row['sed_av']-row['sed_av_16'], row['sed_av_84']-row['sed_av']]]).T
        ha_sed_err = np.array([[row['sed_ha_compare']-row['sed_ha_compare_16'], row['sed_ha_compare_84']-row['sed_ha_compare']]]).T
        pab_sed_err = np.array([[row['sed_pab_compare']-row['sed_pab_compare_16'], row['sed_pab_compare_84']-row['sed_pab_compare']]]).T
        ax_ha.errorbar(ha_flux_jy, row['sed_ha_compare'], yerr=ha_sed_err, ls='None', marker='o', color=rgba, mec='black')
        ax_pab.errorbar(pab_flux_jy, row['sed_pab_compare'], yerr=pab_sed_err, ls='None', marker='o', color=rgba, mec='black')
        ax_lineratio.errorbar(row['emission_fit_lineratio'], row['sed_lineratio'], yerr=sed_av_err, ls='None', marker='o', color=rgba, mec='black')

    if colorbar == True:
        sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar = fig.colorbar(sm, ax=ax_lineratio)
        cbar.set_label(cbar_label, fontsize=16)
        cbar.ax.tick_params(labelsize=16)

    ax_ha.set_title('Halpha')
    ax_pab.set_title('PaBeta')
    ax_lineratio.set_title('Line Ratio')
    for ax in [ax_ha, ax_pab]:
        ax.set_xlabel('Emission Fit Flux')
        ax.set_ylabel('SED Flux')
        ax.set_xscale('log')
        ax.set_yscale('log')
    for ax in axarr:
        ax.tick_params(labelsize=12)
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        ax.plot([-100, 100], [-100, 100], ls='--', color='red', marker='None')
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)

    ax_lineratio.set_xlabel('Emission Fit Line Ratio')
    ax_lineratio.set_ylabel('SED Line Ratio')

    # ax_lineratio.set_xlim(0.8*np.min(lineratio_df['integrated_spec_lineratio']), 1.2*np.max(lineratio_df['integrated_spec_lineratio']))
    # ax_lineratio.set_ylim(0.8*np.min(lineratio_df['sed_lineratio']), 1.2*np.max(lineratio_df['sed_lineratio']))

    # ax_ha.set_xlim(0.8*np.min(lineratio_df['int_spec_ha_compare']), 1.2*np.max(lineratio_df['int_spec_ha_compare']))
    # ax_pab.set_xlim(0.8*np.min(lineratio_df['int_spec_pab_compare']), 1.2*np.max(lineratio_df['int_spec_pab_compare']))
    # ax_ha.set_ylim(0.8*np.min(lineratio_df['sed_ha_compare']), 1.2*np.max(lineratio_df['sed_ha_compare']))
    # ax_pab.set_ylim(0.8*np.min(lineratio_df['sed_pab_compare']), 1.2*np.max(lineratio_df['sed_pab_compare']))

    


    fig.savefig(f'/Users/brianlorenz/uncover/Figures/paper_figures/emission_fit_compare{add_str}.pdf')

if __name__ == "__main__":
    generate_filtered_lineratio_df(aper_size='048')
    # make_av_compare_figure(regenerate=False)
    # intspec_sed_compare()
    # emission_sed_compare()
    pass