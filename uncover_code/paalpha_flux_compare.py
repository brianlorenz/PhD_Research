import matplotlib.pyplot as plt
from astropy.io import ascii
import numpy as np
from compute_av import compute_paalpha_pabeta_av
import pandas as pd
from plot_vals import *
from full_phot_plots import boot_errs


def compare_paa_flux():
    paa_phot_measure_loc = '/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_lineflux_PaAlpha_only.csv'
    paa_phot_measure_df = ascii.read(paa_phot_measure_loc).to_pandas()

    paa_lineflux_calc_loc = '/Users/brianlorenz/uncover/Data/generated_tables/lineflux_df_paalpha_only.csv'
    paa_lineflux_calc_df = ascii.read(paa_lineflux_calc_loc).to_pandas()
    paa_lineflux_calc_df = paa_lineflux_calc_df[paa_lineflux_calc_df['id_msa'] != 18045]

    pab_phot_measure_loc = '/Users/brianlorenz/uncover/Data/generated_tables/phot_calcs/phot_lineflux_PaBeta_only.csv'
    pab_phot_measure_df = ascii.read(pab_phot_measure_loc).to_pandas()


    # Before fixing the bug in make_linemaps, the errors were not being subtracted. Fixing them here
    paa_phot_measure_df['err_PaAlpha_flux_low'] = paa_phot_measure_df['PaAlpha_flux'] - paa_phot_measure_df['err_PaAlpha_flux_low']
    paa_phot_measure_df['err_PaAlpha_flux_high'] = paa_phot_measure_df['err_PaAlpha_flux_high'] - paa_phot_measure_df['PaAlpha_flux']
    pab_phot_measure_df['err_PaBeta_flux_low'] = pab_phot_measure_df['PaBeta_flux'] - pab_phot_measure_df['err_PaBeta_flux_low']
    pab_phot_measure_df['err_PaBeta_flux_high'] = pab_phot_measure_df['err_PaBeta_flux_high'] - pab_phot_measure_df['PaBeta_flux']
    pab_phot_measure_df['err_fe_cor_PaBeta_flux_low'] = pab_phot_measure_df['fe_cor_PaBeta_flux'] - pab_phot_measure_df['err_fe_cor_PaBeta_flux_low']
    pab_phot_measure_df['err_fe_cor_PaBeta_flux_high'] = pab_phot_measure_df['err_fe_cor_PaBeta_flux_high'] - pab_phot_measure_df['fe_cor_PaBeta_flux']


    fig, ax = plt.subplots(figsize=(6,6))

    def plot_column_vs_column(colx, coly, xlabel, ylabel, save_name, title, ids=[], log=True, av=False, xerr=[], yerr=[]):
        if len(yerr)>0:
            ax.errorbar(colx, coly, yerr=yerr, marker='o', ls='None', color='black')
        else:
            ax.plot(colx, coly, marker='o', ls='None', color='black')
        add_str=''
        if len(ids)>0:
            for i in range(len(ids)):
                ax.text(colx.iloc[i], coly.iloc[i], f'{ids[i]}')
            add_str='_label'

        ax.set_xlabel(xlabel, fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(labelsize=14)

        ax.plot([-100, 100], [-100, 100], ls='--', color='red', marker='None')


        min_lim = 0.9*np.min([np.min(np.abs(colx)), np.min(np.abs(coly))])
        max_lim = 1.1*np.max([np.max(colx), np.max(coly)])
        bbox_val = ''
        if av:
            min_lim = 0.9*np.min([np.min(colx), np.min(coly)])
            bbox_val = 'tight'
        ax.set_xlim(min_lim, max_lim)
        ax.set_ylim(min_lim, max_lim)
        
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
            bbox_val = 'tight'
        else:
            ax.set_xscale('linear')
            ax.set_yscale('linear')
        ax.set_title(title)

        fig.savefig(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/{save_name}{add_str}.pdf', bbox_inches=bbox_val)
        plt.close('all')
    
    
    # These make the plots
    err_paalpha_sed_method = pandas_cols_to_matplotlib_errs(paa_phot_measure_df['err_PaAlpha_flux_low'], paa_phot_measure_df['err_PaAlpha_flux_high'])
    # plot_column_vs_column(paa_lineflux_calc_df['paa_emfit_flux'], paa_phot_measure_df['PaAlpha_flux'], 'Emission fit flux', 'SED method flux', 'paalpha_vs_emfit', 'PaAlpha', yerr=err_paalpha_sed_method)
    # plot_column_vs_column(paa_lineflux_calc_df['paa_emfit_flux'], paa_phot_measure_df['PaAlpha_flux'], 'Emission fit flux', 'SED method flux', 'paalpha_vs_emfit', 'PaAlpha', ids=paa_phot_measure_df['id_dr3'], yerr=err_paalpha_sed_method)


    id_pab_paa_list = [15350, 17089, 19283, 25774, 42203, 42238, 48463]
    paa_pab_ratios_emfit = []
    pab_emfit_fluxes = []
    paa_sed_fluxes = []
    for id_msa in id_pab_paa_list:
        paa_lineflux_calc_row = paa_lineflux_calc_df[paa_lineflux_calc_df['id_msa'] == id_msa]
        paa_emfit_flux = paa_lineflux_calc_row['paa_emfit_flux'].iloc[0]

        pab_emfit_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting_pabeta_only/{id_msa}_emission_fits.csv').to_pandas()
        pab_emfit_flux = pab_emfit_df['flux'].iloc[0]
        
        paa_pab_ratio = paa_emfit_flux/pab_emfit_flux
        paa_pab_ratios_emfit.append(paa_pab_ratio)
        pab_emfit_fluxes.append(pab_emfit_flux)

    merged_paa_pab_phot_df = pd.merge(pab_phot_measure_df, paa_phot_measure_df, on='id_dr3', how='inner')
    paa_pab_avs_emfit = [compute_paalpha_pabeta_av(ratio) for ratio in paa_pab_ratios_emfit] 
    pab_phot_measure_df['pab_emfit_flux'] = pab_emfit_fluxes
    pab_phot_measure_df['paa_pab_ratio_emfit'] = paa_pab_ratios_emfit
    merged_paa_pab_phot_df['paa_pab_ratio_emfit'] = paa_pab_ratios_emfit
    pab_phot_measure_df['paa_pab_av_emfit'] = paa_pab_avs_emfit
    merged_paa_pab_phot_df['paa_pab_av_emfit'] = paa_pab_avs_emfit


    
    merged_paa_pab_phot_df['paa_pab_ratio_sed'] = merged_paa_pab_phot_df['PaAlpha_flux'] / merged_paa_pab_phot_df['fe_cor_PaBeta_flux']
    merged_paa_pab_phot_df['paa_pab_av_sed'] = [compute_paalpha_pabeta_av(ratio) for ratio in merged_paa_pab_phot_df['paa_pab_ratio_sed']] 
    err_av_lows = []
    err_av_highs = []
    for i in range(len(merged_paa_pab_phot_df)):
        merged_paa_pab_phot_row = merged_paa_pab_phot_df.iloc[i]
        boot_vals, err_av_low, err_av_high = boot_errs(merged_paa_pab_phot_row['paa_pab_av_sed'], merged_paa_pab_phot_row['PaAlpha_flux'],  np.abs(merged_paa_pab_phot_row['err_PaAlpha_flux_low']),  np.abs(merged_paa_pab_phot_row['err_PaAlpha_flux_high']), merged_paa_pab_phot_row['fe_cor_PaBeta_flux'], np.abs(merged_paa_pab_phot_row['err_fe_cor_PaBeta_flux_low']), np.abs(merged_paa_pab_phot_row['err_fe_cor_PaBeta_flux_high']), paa_pab=True)
        err_av_lows.append(err_av_low)
        err_av_highs.append(err_av_high)
    merged_paa_pab_phot_df['err_paa_pab_av_sed_low'] = err_av_lows
    merged_paa_pab_phot_df['err_paa_pab_av_sed_high'] = err_av_highs
    err_av_sed_method = pandas_cols_to_matplotlib_errs(merged_paa_pab_phot_df['err_paa_pab_av_sed_low'], merged_paa_pab_phot_df['err_paa_pab_av_sed_high'])

    err_pabeta_sed_method = pandas_cols_to_matplotlib_errs(pab_phot_measure_df['err_fe_cor_PaBeta_flux_low'], pab_phot_measure_df['err_fe_cor_PaBeta_flux_high'])
    # plot_column_vs_column(pab_phot_measure_df['pab_emfit_flux'], pab_phot_measure_df['fe_cor_PaBeta_flux'], 'Emission fit flux', 'SED method flux', 'pabeta_vs_emfit', 'PaBeta', yerr=err_pabeta_sed_method)
    # plot_column_vs_column(pab_phot_measure_df['pab_emfit_flux'], pab_phot_measure_df['fe_cor_PaBeta_flux'], 'Emission fit flux', 'SED method flux', 'pabeta_vs_emfit', 'PaBeta', yerr=err_pabeta_sed_method, ids=pab_phot_measure_df['id_dr3'])

    # plot_column_vs_column(merged_paa_pab_phot_df['paa_pab_ratio_emfit'], merged_paa_pab_phot_df['paa_pab_ratio_sed'], 'Emission fit PaA/PaB', 'SED method PaA/PaB', 'paalpha_pabeta_ratio', 'PaA/PaB', log=False)
    # plot_column_vs_column(merged_paa_pab_phot_df['paa_pab_ratio_emfit'], merged_paa_pab_phot_df['paa_pab_ratio_sed'], 'Emission fit PaA/PaB', 'SED method PaA/PaB', 'paalpha_pabeta_ratio', 'PaA/PaB', ids=pab_phot_measure_df['id_dr3'], log=False)
    plot_column_vs_column(merged_paa_pab_phot_df['paa_pab_av_emfit'], merged_paa_pab_phot_df['paa_pab_av_sed'], 'Emission fit PaA/PaB AV', 'SED method PaA/PaB AV', 'paalpha_pabeta_av', 'PaA/PaB AV', log=False, av=True, yerr=err_av_sed_method)
    plot_column_vs_column(merged_paa_pab_phot_df['paa_pab_av_emfit'], merged_paa_pab_phot_df['paa_pab_av_sed'], 'Emission fit PaA/PaB AV', 'SED method PaA/PaB AV', 'paalpha_pabeta_av', 'PaA/PaB AV', ids=pab_phot_measure_df['id_dr3'], log=False, av=True, yerr=err_av_sed_method)



if __name__ == '__main__':

    compare_paa_flux()