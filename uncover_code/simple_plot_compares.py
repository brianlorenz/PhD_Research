import matplotlib.pyplot as plt
import matplotlib as mpl
from astropy.io import ascii
from plot_vals import scale_aspect
from uncover_read_data import read_lineflux_cat, get_id_msa_list, read_SPS_cat
from compute_av import compute_ha_pab_av
import numpy as np

def paper_plot_sed_emfit_accuracy(id_msa_list, color_var=''):
    full_data_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df.csv').to_pandas()
    data_df = full_data_df[full_data_df['id_msa'].isin(id_msa_list)]
    full_lineratio_data_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineratio_av_df.csv').to_pandas()
    lineratio_data_df = full_lineratio_data_df[full_lineratio_data_df['id_msa'].isin(id_msa_list)]
    sps_df = read_SPS_cat()

    fig, axarr = plt.subplots(1,3,figsize=(18,6))
    ax_ha_sed_vs_emfit = axarr[0]
    ax_pab_sed_vs_emfit = axarr[1]
    ax_av_sed_vs_emfit = axarr[2]
    ax_list = [ax_ha_sed_vs_emfit, ax_pab_sed_vs_emfit]

    markersize=8

    for id_msa in id_msa_list:
        data_df_row = data_df[data_df['id_msa'] == id_msa]
        lineratio_data_row = lineratio_data_df[lineratio_data_df['id_msa'] == id_msa]
        sps_row = sps_df[sps_df['id_msa']==id_msa]
        fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
        ha_snr = fit_df['signal_noise_ratio'].iloc[0]
        pab_snr = fit_df['signal_noise_ratio'].iloc[1]
        
        cmap = mpl.cm.inferno
        
        if color_var == 'sed_av':
            norm = mpl.colors.Normalize(vmin=0, vmax=3) 
            rgba = cmap(norm(lineratio_data_row['sed_av']))  
            cbar_label = 'Photometry AV'
        if color_var == 'ha_snr':
            norm = mpl.colors.LogNorm(vmin=2, vmax=100) 
            rgba = cmap(norm(ha_snr))
            cbar_label = 'H$\\alpha$ SNR'
        if color_var == 'pab_snr':
            norm = mpl.colors.Normalize(vmin=5, vmax=20) 
            rgba = cmap(norm(pab_snr))
            cbar_label = 'Pa$\\beta$ SNR'
        elif color_var == 'metallicity':
            norm = mpl.colors.Normalize(vmin=-1.2, vmax=0.1) 
            rgba = cmap(norm(sps_row['met_50']))
            cbar_label = 'Prospector Metallicity'
        if color_var != 'None':
            color_str = f'_{color_var}'
        else:
            color_str = ''

        # Emission fit data
        fit_nii_cor_ha_flux = fit_df['nii_cor_flux'].iloc[0]
        err_fit_nii_cor_ha_flux_low = fit_df['err_nii_cor_flux_low'].iloc[0]
        err_fit_nii_cor_ha_flux_high = fit_df['err_nii_cor_flux_high'].iloc[0]
        fit_pab_flux = fit_df['nii_cor_flux'].iloc[1]
        err_fit_pab_flux_low = fit_df['err_nii_cor_flux_low'].iloc[1]
        err_fit_pab_flux_high = fit_df['err_nii_cor_flux_high'].iloc[1]

        err_sed_nii_cor_ha_flux_low = lineratio_data_row['err_nii_cor_sed_ha_lineflux_low'].iloc[0]
        err_sed_nii_cor_ha_flux_high = lineratio_data_row['err_nii_cor_sed_ha_lineflux_high'].iloc[0]
        err_sed_fe_cor_pab_flux_low = lineratio_data_row['err_fe_cor_sed_pab_lineflux_low'].iloc[0]
        err_sed_fe_cor_pab_flux_high = lineratio_data_row['err_fe_cor_sed_pab_lineflux_high'].iloc[0]

        ax_ha_sed_vs_emfit.errorbar(fit_nii_cor_ha_flux, data_df_row['nii_cor_ha_sed_flux'], xerr=[[err_fit_nii_cor_ha_flux_low], [err_fit_nii_cor_ha_flux_high]], yerr=[[err_sed_nii_cor_ha_flux_low],[err_sed_nii_cor_ha_flux_high]], marker='o', color=rgba, ls='None', mec='black', ms=markersize, ecolor='black')
        ax_ha_sed_vs_emfit.set_xlabel('H$\\alpha$ Spectrum')
        ax_ha_sed_vs_emfit.set_ylabel('H$\\alpha$ Photometry')

        ax_pab_sed_vs_emfit.errorbar(fit_pab_flux, data_df_row['fe_cor_pab_sed_flux'], xerr=[[err_fit_pab_flux_low], [err_fit_pab_flux_high]], yerr=[[err_sed_fe_cor_pab_flux_low],[err_sed_fe_cor_pab_flux_high]], marker='o', color=rgba, ls='None', mec='black', ms=markersize, ecolor='black')
        ax_pab_sed_vs_emfit.set_xlabel('Pa$\\beta$ Spectrum')
        ax_pab_sed_vs_emfit.set_ylabel('Pa$\\beta$ Photometry')

        ax_av_sed_vs_emfit.errorbar(lineratio_data_row['emission_fit_av'], lineratio_data_row['sed_av'], xerr=[[lineratio_data_row['err_emission_fit_av_low'].iloc[0]], [lineratio_data_row['err_emission_fit_av_high'].iloc[0]]], yerr=[[lineratio_data_row['err_sed_av_low'].iloc[0]], [lineratio_data_row['err_sed_av_high'].iloc[0]]], marker='o', color=rgba, ls='None', mec='black', ms=markersize, ecolor='black')
        ax_av_sed_vs_emfit.set_xlabel('AV Spectrum')
        ax_av_sed_vs_emfit.set_ylabel('AV Photometry')
    
        add_text = 1
        if add_text:
            ax_av_sed_vs_emfit.text(lineratio_data_row['emission_fit_av'], lineratio_data_row['sed_av'], f'{id_msa}')
            ax_pab_sed_vs_emfit.text(fit_pab_flux, data_df_row['fe_cor_pab_sed_flux'], f'{id_msa}')
            ax_ha_sed_vs_emfit.text(fit_nii_cor_ha_flux, data_df_row['nii_cor_ha_sed_flux'], f'{id_msa}')


    for ax in ax_list:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=12)
        ax.plot([1e-20, 1e-15], [1e-20, 1e-15], ls='--', color='red', marker='None')
        ax.set_xlim([5e-19, 8e-16])
        ax.set_ylim([5e-19, 8e-16])
        scale_aspect(ax)
    ax_av_sed_vs_emfit.tick_params(labelsize=12)
    ax_av_sed_vs_emfit.plot([-10, 100], [-10, 100], ls='--', color='red', marker='None')
    ax_av_sed_vs_emfit.set_xlim([-1.5, 4.5])
    ax_av_sed_vs_emfit.set_ylim([-1.5, 4.5])
    scale_aspect(ax_av_sed_vs_emfit)

    cb_ax = fig.add_axes([.91,.124,.02,.734])
    if color_var != 'None':
        sm =  mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        cbar_ticks = [5, 10, 15, 20]
        cbar_ticklabels = [str(tick) for tick in cbar_ticks]
        cbar = fig.colorbar(sm, orientation='vertical', cax=cb_ax, ticks=cbar_ticks)
        cbar.ax.set_yticklabels(cbar_ticklabels)  
        cbar.set_label(cbar_label, fontsize=16)
        cbar.ax.tick_params(labelsize=16)

    save_loc = f'/Users/brianlorenz/uncover/Figures/paper_figures/sed_vs_emfit{color_str}.pdf'
    fig.savefig(save_loc)


def plot_simpletests(id_msa_list):
    full_data_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df.csv').to_pandas()
    data_df = full_data_df[full_data_df['id_msa'].isin(id_msa_list)]
    full_av_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineratio_av_df.csv').to_pandas()
    av_df = full_av_df[full_av_df['id_msa'].isin(id_msa_list)]

    fig, axarr = plt.subplots(2,4,figsize=(22,10))
    ax_ha_sed_vs_cat = axarr[0,0]
    ax_ha_sed_vs_emfit = axarr[0,1]
    ax_ha_cat_vs_emfit = axarr[0,2]
    ax_pab_sed_vs_cat = axarr[1,0]
    ax_pab_sed_vs_emfit = axarr[1,1]
    ax_pab_cat_vs_emfit = axarr[1,2]
    ax_ratio_cat_vs_emfit = axarr[0,3]
    ax_list = [ax_ha_sed_vs_cat, ax_ha_sed_vs_emfit, ax_ha_cat_vs_emfit, ax_pab_sed_vs_cat, ax_pab_sed_vs_emfit, ax_pab_cat_vs_emfit]

    sed_label = 'SED method'
    cat_label = 'UNCOVER Catalog'
    emfit_label = 'emission fit'

    ax_ha_sed_vs_cat.plot(data_df['ha_cat_flux'], data_df['ha_sed_flux'], marker='o', color='black', ls='None')
    ax_ha_sed_vs_cat.set_xlabel(cat_label)
    ax_ha_sed_vs_cat.set_ylabel(sed_label)

    ax_ha_sed_vs_emfit.plot(data_df['ha_emfit_flux'], data_df['ha_sed_flux'], marker='o', color='black', ls='None')
    ax_ha_sed_vs_emfit.set_xlabel(emfit_label)
    ax_ha_sed_vs_emfit.set_ylabel(sed_label)

    ax_ha_cat_vs_emfit.plot(data_df['ha_cat_flux'], data_df['ha_emfit_flux'], marker='o', color='black', ls='None')
    ax_ha_cat_vs_emfit.set_xlabel(cat_label)
    ax_ha_cat_vs_emfit.set_ylabel(emfit_label)

    ax_pab_sed_vs_cat.plot(data_df['pab_cat_flux'], data_df['pab_sed_flux'], marker='o', color='black', ls='None')
    ax_pab_sed_vs_cat.set_xlabel(cat_label)
    ax_pab_sed_vs_cat.set_ylabel(sed_label)

    ax_pab_sed_vs_emfit.plot(data_df['pab_emfit_flux'], data_df['pab_sed_flux'], marker='o', color='black', ls='None')
    ax_pab_sed_vs_emfit.set_xlabel(emfit_label)
    ax_pab_sed_vs_emfit.set_ylabel(sed_label)

    ax_pab_cat_vs_emfit.plot(data_df['pab_cat_flux'], data_df['pab_emfit_flux'], marker='o', color='black', ls='None')
    ax_pab_cat_vs_emfit.set_xlabel(cat_label)
    ax_pab_cat_vs_emfit.set_ylabel(emfit_label)

    cat_ratio = data_df['ha_cat_flux'] / data_df['pab_cat_flux']
    cat_av = compute_ha_pab_av(1/cat_ratio)
    emfit_ratio = data_df['ha_emfit_flux'] / data_df['pab_emfit_flux']
    emfit_av = compute_ha_pab_av(1/emfit_ratio)

    ax_ratio_cat_vs_emfit.plot(emfit_av, cat_av, marker='o', color='black', ls='None')
    for i in range(len(data_df)):
        if cat_av.iloc[i] > -2.5:
            ax_ha_cat_vs_emfit.text(data_df['ha_cat_flux'].iloc[i], data_df['ha_emfit_flux'].iloc[i], data_df['id_msa'].iloc[i])
            ax_pab_cat_vs_emfit.text(data_df['pab_cat_flux'].iloc[i], data_df['pab_emfit_flux'].iloc[i], data_df['id_msa'].iloc[i])
            ax_ratio_cat_vs_emfit.text(emfit_av.iloc[i], cat_av.iloc[i], data_df['id_msa'].iloc[i])

    ax_ratio_cat_vs_emfit.set_ylabel('Catalog Lineflux AV')
    ax_ratio_cat_vs_emfit.set_xlabel('Emission Fit AV')
    ax_ratio_cat_vs_emfit.tick_params(labelsize=12)
    ax_ratio_cat_vs_emfit.plot([-100, 100], [-100, 100], ls='--', color='red', marker='None')
    ax_ratio_cat_vs_emfit.set_xlim(-2.5, 2.5)
    ax_ratio_cat_vs_emfit.set_ylim(-2.5, 2.5)
    scale_aspect(ax_ratio_cat_vs_emfit)


    for ax in ax_list:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=12)
        ax.plot([1e-20, 1e-15], [1e-20, 1e-15], ls='--', color='red', marker='None')
        ax.set_xlim([1e-20, 1e-15])
        ax.set_ylim([1e-20, 1e-15])
        scale_aspect(ax)
        
    plt.tight_layout()
    save_loc = '/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/simpletest_flux_compare.pdf'
    fig.savefig(save_loc)

def plot_snr_compare(id_msa_list):
    fig, axarr = plt.subplots(1,2,figsize=(12,8))
    ax_ha_snr = axarr[0]
    ax_pab_snr = axarr[1]
    ax_list = [ax_ha_snr, ax_pab_snr]

    lines_df = read_lineflux_cat()


    for id_msa in id_msa_list:
        fit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{id_msa}_emission_fits.csv').to_pandas()
        ha_flux_fit = fit_df.iloc[0]['flux']
        pab_flux_fit = fit_df.iloc[1]['flux']
        ha_sigma = fit_df.iloc[0]['sigma'] # full width of the line
        pab_sigma = fit_df.iloc[1]['sigma'] # full width of the line
        ha_snr = fit_df['signal_noise_ratio'].iloc[0]
        pab_snr = fit_df['signal_noise_ratio'].iloc[1]

        lines_df_row = lines_df[lines_df['id_msa'] == id_msa]
        lines_df_ha_snr = lines_df_row['f_Ha+NII'].iloc[0] / lines_df_row['e_Ha+NII'].iloc[0]
        lines_df_pab_snr = lines_df_row['f_PaB'].iloc[0] / lines_df_row['e_PaB'].iloc[0]

        print(ha_snr)
    
        ax_ha_snr.plot(lines_df_ha_snr, ha_snr, marker='o', color='black', ls='None')
        ax_pab_snr.plot(lines_df_pab_snr, pab_snr, marker='o', color='black', ls='None')
    
    ax_ha_snr.set_xlabel('Catalog Ha SNR')
    ax_ha_snr.set_ylabel('Emfit Ha SNR')

    ax_pab_snr.set_xlabel('Catalog PaB SNR')
    ax_pab_snr.set_ylabel('Emfit PaB SNR')


    for ax in ax_list:
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=12)
        ax.plot([-2, 10000], [-2, 10000], ls='--', color='red', marker='None')
        ax.set_xlim([0.01, 500])
        ax.set_ylim([0.01, 500])
        scale_aspect(ax)
        
    plt.tight_layout()
    save_loc = '/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/snr_compare_cat_emfit.pdf'
    fig.savefig(save_loc)

def plot_offsets(all=False):
    data_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df.csv')
    if all:
        data_df = ascii.read(f'/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df.csv')

    fig, ax = plt.subplots(1,1,figsize=(6,6))
   
    # sed_label = 'SED method'
    # cat_label = 'UNCOVER Catalog'
    # emfit_label = 'emission fit'

    ax.plot(data_df['ha_sed_div_emfit'], data_df['pab_sed_div_emfit'], marker='o', color='black', ls='None')
    ax.set_xlabel('Ha offset sed/emfit')
    ax.set_ylabel('PaB offset sed/emfit')

    


    # for ax in ax_list:
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(labelsize=12)
    ax.plot([-10, 100], [-10, 100], ls='--', color='red', marker='None')
    ax.set_xlim([0.5, 40])
    ax.set_ylim([0.5, 40])
    
        
    plt.tight_layout()
    save_loc = '/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df.pdf'
    if all:
        save_loc = '/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df.pdf'
    fig.savefig(save_loc)

if __name__ == "__main__":
    id_msa_list = get_id_msa_list(full_sample=False)
    paper_plot_sed_emfit_accuracy(id_msa_list, color_var='pab_snr')
    # plot_simpletests(id_msa_list)
    # plot_offsets(all=True)

    # id_msa_list = get_id_msa_list(full_sample=True)

    # plot_snr_compare(id_msa_list)