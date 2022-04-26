# Plot metallicity vs axis ratio for each group


from re import L
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.patheffects as path_effects
from matplotlib.gridspec import GridSpec
from brokenaxes import brokenaxes
from sympy import symbols, solve



def measure_metals(n_groups, save_name, bootstrap=-1):
    """Make the metallicity measurements of the axis groups using the methods of Bian 2018
    
    Parameters:
    n_groups (int): Number of axis ratio groups
    save_name (str): Folder where results are stored
    bootsrap (int): Set to -1 to skip, otherwise set to number of bootstraps
    """

    ### OIII Hbeta method
    axis_groups = []
    O3_Hb_metals = []
    err_O3_Hb_metals = []
    log_03_Hbs = []
    err_log_03_Hbs = []
    log_N2_Ha_measures = []
    err_log_N2_Ha_measures = []
    N2_Ha_metals = []
    err_N2_Ha_metals = []
    log_O3N2_measures = []
    err_log_O3N2_measures = []
    O3N2_metals = []
    err_O3N2_metals = []
    boot_err_O3N2_metal_lows = []
    boot_err_O3N2_metal_highs = []


    for axis_group in range(n_groups):
        fit_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_fits/{axis_group}_emission_fits.csv').to_pandas()
        O3_5008_row = np.argmin(np.abs(fit_df['line_center_rest'] - 5008))
        O3_4960_row = np.argmin(np.abs(fit_df['line_center_rest'] - 4960))
        Hb_row = np.argmin(np.abs(fit_df['line_center_rest'] - 4862))
        Ha_row = np.argmin(np.abs(fit_df['line_center_rest'] - 6563))
        N2_6585_row = np.argmin(np.abs(fit_df['line_center_rest'] - 6585))


        def compute_err_and_logerr(numerator, denominator, err_numerator, err_denominator):
            """
            Parameters:
            numerator (float): Top part of fraction
            denominator (float): bottom part
            err_numerator (float): Uncertainty in the numerator
            err_denominator (float): same for denominator

            Returns:
            result (float): Numerator/Denominator
            log_result (float): np.log10(result)
            err_result (float): Uncertainty in result
            err_log_result (float): Uncertainty in np.log10(result)
            """

            result = numerator/denominator
            log_result = np.log10(result)
            err_result = result * np.sqrt((err_numerator / numerator)**2 + (err_denominator / denominator)**2)
            err_log_result = 0.434294 * (err_result / result)
            return result, log_result, err_result, err_log_result

        O3_numerator = fit_df.iloc[O3_5008_row]['flux']+fit_df.iloc[O3_4960_row]['flux']
        err_O3_numerator = np.sqrt(fit_df.iloc[O3_5008_row]['err_flux']**2+fit_df.iloc[O3_4960_row]['err_flux']**2)
        O3_Hb_measure, log_O3_Hb_measure, err_O3_Hb_measure, err_log_O3_Hb_measure = compute_err_and_logerr(O3_numerator, fit_df.iloc[Hb_row]['flux'], err_O3_numerator, fit_df.iloc[Hb_row]['err_flux'])
        

        N2_Ha_measure, log_N2_Ha_measure, err_N2_Ha_measure, err_log_N2_Ha_measure = compute_err_and_logerr(fit_df.iloc[N2_6585_row]['flux'], fit_df.iloc[Ha_row]['flux'], fit_df.iloc[N2_6585_row]['err_flux'], fit_df.iloc[Ha_row]['err_flux'])
        # N2_Ha_measure = (fit_df.iloc[N2_6585_row]['flux']) / fit_df.iloc[Ha_row]['flux']
        # log_N2_Ha_measure = np.log10(N2_Ha_measure)

        O3N2_numerator, log_O3N2_numerator, err_O3N2_numerator, err_log_O3N2_numerator = compute_err_and_logerr(fit_df.iloc[O3_5008_row]['flux'], fit_df.iloc[Hb_row]['flux'], fit_df.iloc[O3_5008_row]['err_flux'], fit_df.iloc[Hb_row]['err_flux'])
        O3N2_measure, log_O3N2_measure, err_O3N2_measure, err_log_O3N2_measure = compute_err_and_logerr(O3N2_numerator, N2_Ha_measure, err_O3N2_numerator, err_N2_Ha_measure)

        def compute_O3N2_metallicity(log_O3N2_measure, err_log_O3N2_measure):
            """From Bian 2018"""
            O3N2_metal = 8.97 - 0.39*log_O3N2_measure
            err_O3N2_metal = 0.32*err_log_O3N2_measure
            return O3N2_metal, err_O3N2_metal


        # Bian 2018 O3N2 metallicity
        x = symbols('x')
        expr = -log_O3_Hb_measure + 43.9836 - 21.6211*(x) + 3.4277*(x**2) - 0.1747*(x**3)
        err_expr = -log_O3_Hb_measure + err_log_O3_Hb_measure + 43.9836 - 21.6211*(x) + 3.4277*(x**2) - 0.1747*(x**3)
        sol = solve(expr)
        err_sol = solve(err_expr)
        O3_Hb_metal = np.real(complex(sol[2]))
        err_O3_Hb_metal = np.abs(np.real(complex(err_sol[2]))-O3_Hb_metal)

        # PP04 N2 metallicity
        N2_Ha_metal = 8.9 + 0.57*log_N2_Ha_measure
        err_N2_Ha_metal = err_log_N2_Ha_measure*0.57
        N2_Ha_metals.append(N2_Ha_metal)
        err_N2_Ha_metals.append(err_N2_Ha_metal)
        log_N2_Ha_measures.append(log_N2_Ha_measure)
        err_log_N2_Ha_measures.append(err_log_N2_Ha_measure)

        # PP04 O3N2 metallicity
        # O3N2_metal = 8.73 - 0.32*log_O3N2_measure
        # err_O3N2_metal = 0.32*err_log_O3N2_measure
        # Bian 2018 version of O3N2
        O3N2_metal, err_O3N2_metal = compute_O3N2_metallicity(log_O3N2_measure, err_log_O3N2_measure)
        O3N2_metals.append(O3N2_metal)
        err_O3N2_metals.append(err_O3N2_metal)
        log_O3N2_measures.append(log_O3N2_measure)
        err_log_O3N2_measures.append(err_log_O3N2_measure)

        axis_groups.append(axis_group)
        O3_Hb_metals.append(O3_Hb_metal)
        err_O3_Hb_metals.append(err_O3_Hb_metal)
        log_03_Hbs.append(log_O3_Hb_measure)
        err_log_03_Hbs.append(err_log_O3_Hb_measure)


        if bootstrap > -1:
            boot_O3N2s = []
            for bootstrap_num in range(bootstrap):
                boot_fit_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_fits_boots/{axis_group}_emission_fits_{bootstrap_num}.csv').to_pandas()
                boot_N2_Ha_measure, boot_log_N2_Ha_measure, boot_err_N2_Ha_measure, boot_err_log_N2_Ha_measure = compute_err_and_logerr(boot_fit_df.iloc[N2_6585_row]['flux'], boot_fit_df.iloc[Ha_row]['flux'], boot_fit_df.iloc[N2_6585_row]['err_flux'], boot_fit_df.iloc[Ha_row]['err_flux'])
                boot_O3N2_numerator, boot_log_O3N2_numerator, boot_err_O3N2_numerator, boot_err_log_O3N2_numerator = compute_err_and_logerr(boot_fit_df.iloc[O3_5008_row]['flux'], boot_fit_df.iloc[Hb_row]['flux'], boot_fit_df.iloc[O3_5008_row]['err_flux'], boot_fit_df.iloc[Hb_row]['err_flux'])
                boot_O3N2_measure, boot_log_O3N2_measure, boot_err_O3N2_measure, boot_err_log_O3N2_measure = compute_err_and_logerr(boot_O3N2_numerator, boot_N2_Ha_measure, boot_err_O3N2_numerator, boot_err_N2_Ha_measure)
                boot_O3N2_metal, _ = compute_O3N2_metallicity(boot_log_O3N2_measure, boot_err_log_O3N2_measure)
                boot_O3N2s.append(boot_O3N2_metal)
            boot_err_O3N2_metal_low = O3N2_metal - np.percentile(boot_O3N2s, 16)
            boot_err_O3N2_metal_high = np.percentile(boot_O3N2s, 84) - O3N2_metal
            boot_err_O3N2_metal_lows.append(boot_err_O3N2_metal_low)
            boot_err_O3N2_metal_highs.append(boot_err_O3N2_metal_high)


    metals_df = pd.DataFrame(zip(axis_groups, log_03_Hbs, err_log_03_Hbs, O3_Hb_metals, err_O3_Hb_metals, log_N2_Ha_measures, err_log_N2_Ha_measures, N2_Ha_metals, err_N2_Ha_metals, log_O3N2_measures, err_log_O3N2_measures, O3N2_metals, err_O3N2_metals, boot_err_O3N2_metal_lows, boot_err_O3N2_metal_highs), columns=['axis_group', 'log_03_Hb_measure', 'err_log_03_Hb_measure', 'O3_Hb_metallicity', 'err_O3_Hb_metallicity', 'log_N2_Ha_measure', 'err_log_N2_Ha_measure', 'N2_Ha_metallicity', 'err_N2_Ha_metallicity', 'log_O3N2_measure', 'err_log_O3N2_measure', 'O3N2_metallicity', 'err_O3N2_metallicity', 'boot_err_O3N2_metallicity_low', 'boot_err_O3N2_metallicity_high'])
    metals_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/group_metallicities.csv', index=False)


def plot_group_metals_compare(n_groups, save_name):
    fig, axarr = plt.subplots(2, 2, figsize=(8,8))
    ax_Bian_N2 = axarr[0, 0]
    ax_O3N2_N2 = axarr[1, 0]
    ax_O3N2_Bian = axarr[1, 1]

    ax_Bian_N2.set_xlabel('N2 Metallicity', fontsize=14)
    ax_Bian_N2.set_ylabel('Bian O3Hb Metallicity', fontsize=14)
    ax_O3N2_N2.set_xlabel('N2 Metallicity', fontsize=14)
    ax_O3N2_N2.set_ylabel('O3N2 Metallicity', fontsize=14)
    ax_O3N2_Bian.set_xlabel('Bian O3Hb Metallicity', fontsize=14)
    ax_O3N2_Bian.set_ylabel('O3N2 Metallicity', fontsize=14)

    metals_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/group_metallicities.csv').to_pandas()
    for i in range(n_groups):
        Bian_metal = metals_df['O3_Hb_metallicity'].iloc[i]
        err_Bian_metal = metals_df['err_O3_Hb_metallicity'].iloc[i]
        N2_metal = metals_df['N2_Ha_metallicity'].iloc[i]
        err_N2_metal = metals_df['err_N2_Ha_metallicity'].iloc[i]
        O3N2_metal = metals_df['O3N2_metallicity'].iloc[i]
        err_O3N2_metal = metals_df['err_O3N2_metallicity'].iloc[i]
        ax_Bian_N2.errorbar(N2_metal, Bian_metal, xerr=err_N2_metal, yerr=err_Bian_metal, marker='o', ls='None', color='black')
        ax_O3N2_N2.errorbar(N2_metal, O3N2_metal, xerr=err_N2_metal, yerr=err_O3N2_metal, marker='o', ls='None', color='black')
        ax_O3N2_Bian.errorbar(Bian_metal, O3N2_metal, xerr=err_Bian_metal, yerr=err_O3N2_metal, marker='o', ls='None', color='black')

    for ax in [ax_Bian_N2, ax_O3N2_N2, ax_O3N2_Bian]:
        ax.set_xlim(8.1, 8.9)
        ax.set_ylim(8.1, 8.9)
        ax.plot((8, 9.5), (8, 9.5), ls='--', color='red', marker='None')

    axarr[0, 1].axis('off')

    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/group_metallicity_compare.pdf')



def plot_metals(savename):
    """Make the plot of individual galaxy metallicities
    
    """
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{savename}/summary.csv').to_pandas()
    

    n_rows = int(len(summary_df) / 6)
    if len(summary_df) == 8:
        n_rows = 2
    if len(summary_df) == 4:
        n_rows = 2
    if savename=='both_sfms_6bin_median_2axis':
        n_rows=3
    if savename=='both_sfms_6bin_median_1axis':
        n_rows=3
    if savename=='both_6bin_1axis_median_params':
        n_rows=3
    
    fig, axarr = plt.subplots(n_rows, 2, figsize=(12, 6+3*n_rows))
        

    if n_rows == 1:
        ax_0 = axarr[0,0]
        ax_1 = axarr[0,1]
    if n_rows > 1:
        ax_0 = axarr[1,0]
        ax_1 = axarr[0,0]
        ax_2 = axarr[1,1]
        ax_3 = axarr[0,1]
    if n_rows > 2:
        ax_0 = axarr[2,0]
        ax_1 = axarr[1,0]
        ax_2 = axarr[0,0]
        ax_3 = axarr[2,1]
        ax_4 = axarr[1,1]
        ax_5 = axarr[0,1]

    # plot_lims = ((4850, 4875), (6540, 6590))

        
    # bax_lowm_highs = brokenaxes(xlims=plot_lims, subplot_spec=axarr[0,0])
    # bax_highm_highs = brokenaxes(xlims=plot_lims, subplot_spec=axarr[0,1])
    # bax_lowm_lows = brokenaxes(xlims=plot_lims, subplot_spec=axarr[1,0])
    # bax_highm_lows = brokenaxes(xlims=plot_lims, subplot_spec=axarr[1,1])
    # bax_lowest_lows = brokenaxes(xlims=plot_lims, subplot_spec=axarr[2,0])
    # bax_lowest_highs = brokenaxes(xlims=plot_lims, subplot_spec=axarr[2,1])


    for i in range(len(summary_df)):
        row = summary_df.iloc[i]
        axis_group = row['axis_group']

        if row['key'] == 'sorted0':
            ax = ax_0
            ax.set_title('sorted0', color = row['color'], fontsize=14)
        if row['key'] == 'sorted1':
            ax = ax_1
            ax.set_title('sorted1', color = row['color'], fontsize=14)
        if row['key'] == 'sorted2':
            ax = ax_2
            ax.set_title('sorted2', color = row['color'], fontsize=14)
        if row['key'] == 'sorted3':
            ax = ax_3
            ax.set_title('sorted3', color = row['color'], fontsize=14)
        if row['key'] == 'sorted4':
            ax = ax_4
            ax.set_title('sorted4', color = row['color'], fontsize=14)
        if row['key'] == 'sorted5':
            ax = ax_5
            ax.set_title('sorted5', color = row['color'], fontsize=14)
        
        ar_df = ascii.read(imd.axis_cluster_data_dir + f'/{savename}/{savename}_group_dfs/{axis_group}_df.csv').to_pandas()


        if row['shape'] == '+': 
            color = 'red'
            label = 'Axis Ratio < 0.4'
        if row['shape'] == 'd':
            color = 'mediumseagreen'
            label = '0.4 < Axis Ratio < 0.7'
        if row['shape'] == 'o':
            color = 'blue'
            label = '0.7 < Axis Ratio'
        if row['shape'] == 1.0: 
            color = 'red'
            label = ''

        # compute uncertainties
        ar_df['metal_err_high'] = ar_df['u68_logoh_pp_n2'] - ar_df['logoh_pp_n2']
        ar_df['metal_err_low'] = ar_df['logoh_pp_n2'] - ar_df['l68_logoh_pp_n2']  
        
       
        # Plot 
        upper_lim_idx = ar_df['n2flag_metals'] == 1

        # Errors on non-lower-lim points
        yerrs_low = [ar_df['metal_err_low'][~upper_lim_idx].iloc[k] for k in range(len(ar_df[~upper_lim_idx]))]
        yerrs_high = [ar_df['metal_err_high'][~upper_lim_idx].iloc[k] for k in range(len(ar_df[~upper_lim_idx]))]
        yerrs = (yerrs_low, yerrs_high)

        ax.errorbar(ar_df['use_ratio'][~upper_lim_idx], ar_df['logoh_pp_n2'][~upper_lim_idx], xerr=ar_df['err_use_ratio'][~upper_lim_idx], yerr = yerrs, color=color, label = label, marker='o', ls='None') 
        ax.errorbar(ar_df['use_ratio'][upper_lim_idx], ar_df['logoh_pp_n2'][upper_lim_idx], xerr=ar_df['err_use_ratio'][upper_lim_idx], color=color, label = label, marker='o', mfc='white', ls='None') 
        ax.set_ylim(8, 9)
        ax.set_ylabel('12 + log(O/H)', fontsize=14)
        ax.set_xlabel('Axis Ratio', fontsize=14)


        if i == len(summary_df)-1:
            ax.legend(bbox_to_anchor=(1.5, 4.5, 0.20, 0.15), loc='upper right')

        ax.tick_params(labelsize=12, size=8)

    fig.savefig(imd.axis_cluster_data_dir + f'/{savename}/metallicty_ar.pdf')
    plt.close('all')

def plot_mass_metal(n_groups, save_name):
    fig, ax = plt.subplots(figsize=(8,8))

    ax.set_xlabel('log(Stellar Mass)', fontsize=14)
    ax.set_ylabel('O3N2 Metallicity', fontsize=14)


    metals_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/group_metallicities.csv').to_pandas()
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv').to_pandas()
    for i in range(n_groups):
        O3N2_metal = metals_df['O3N2_metallicity'].iloc[i]
        err_O3N2_metal = metals_df['err_O3N2_metallicity'].iloc[i]
        mass = summary_df['log_mass_median'].iloc[i]

        ax.errorbar(mass, O3N2_metal, yerr=err_O3N2_metal, marker='o', ls='None', color='black')
       
    ax.set_xlim(9, 11)
    ax.set_ylim(8.1, 8.9)


    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/mass_metallicity.pdf')


def add_metals_to_summary_df(save_name, metal_column):
    metals_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/group_metallicities.csv').to_pandas()
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv').to_pandas()

    # Have to name it this way for rest of code to recognize it as a color_var
    summary_df['metallicity_median'] = metals_df[metal_column]
    summary_df['err_metallicity_median'] = metals_df['err_'+metal_column]
    summary_df['err_metallicity_median_low'] = metals_df['boot_err_'+metal_column+'_low']
    summary_df['err_metallicity_median_high'] = metals_df['boot_err_'+metal_column+'_high']

    summary_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv', index=False)

# plot_metals(savename='halpha_ssfr_4bin_mean_shifted')
# measure_metals(8, 'both_sfms_4bin_median_2axis_boot100', bootstrap=100)
# plot_group_metals_compare(12, 'both_ssfrs_4bin_mean')
# plot_mass_metal(12, 'both_ssfrs_4bin_mean')
# add_metals_to_summary_df('both_sfms_4bin_median_2axis_boot100', metal_column='O3N2_metallicity')