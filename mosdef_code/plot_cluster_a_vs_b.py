import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot_vals import *
from leja_sfms_redshift import leja2022_sfms
from astropy.io import ascii
import os
import pandas as pd
import sys
if os.path.exists(imd.loc_cluster_summary_df): 
    cluster_summary_df = imd.read_cluster_summary_df()

def plot_cluster_summaries(x_var, y_var, savename, color_var='None', plot_lims='None', one_to_one=False, ignore_groups=[], log=False, lower_limit=False, add_leja=False, yerr=False, prospector_run_name = ''):
    """Plots two columsn of cluster_summary_df against each other
    
    Parameters:
    x_var (str): x variable to plot
    y_var (str): y variable to plot
    savename (str): Name to save the figure under
    color_var (str): 'None' or the column to use as color
    plot_lims (list of 4): [xmin, xmax, ymin, ymax]
    one_to_one (boolean): Set to True to add a 1-1 line
    log (boolean): Set to True to make it a log-log plot
    lower_limit (boolean): Set to true to use lower limit SFRs and hollow out those points
    add_leja (boolean): If True, add the sfms from Leja 2022 to the plot
    yerr (boolean): Set to true to plot errorbars on the yaxis
    prospector_run_name (str): Set to the prospector run name if using
    """

    fig, ax = plt.subplots(figsize = (8,8))

    if lower_limit == True:
        y_var = y_var+'_with_limit'
        savename = savename+'_with_limit'


    if y_var == 'override_flux_with_limit':
        cluster_summary_df2 = ascii.read(imd.mosdef_dir + '/Clustering_20230823_scaledtoindivha/cluster_summary.csv').to_pandas()
        cluster_summary_df['override_flux_with_limit'] = cluster_summary_df2[x_var]

    for i in range(len(cluster_summary_df)):
        if i in ignore_groups:
            continue
        row = cluster_summary_df.iloc[i]

        if prospector_run_name != '':
            if os.path.exists(imd.prospector_emission_fits_dir + f'/{prospector_run_name}_emission_fits/{i}_emission_fits.csv'):
                prospector_emission = ascii.read(imd.prospector_emission_fits_dir + f'/{prospector_run_name}_emission_fits/{i}_emission_fits.csv').to_pandas()
                ha_row = prospector_emission[prospector_emission['line_name']=='Halpha']
                row[y_var] = ha_row[y_var]
            else:
                continue

        if color_var != 'None':
            cmap = mpl.cm.inferno
            norm = assign_color(color_var)
            rgba = cmap(norm(row[color_var]))
        else:
            rgba = 'black'

        # Make the point hollow if it's a lower limit
        if lower_limit == True:
            if row['flag_balmer_lower_limit']==1:
                marker='^'
            else:
                marker='o'
        else:
            marker='o'
        if yerr == True:
            try:
                ax.errorbar(row[x_var], row[y_var], yerr=np.array([[row['err_'+y_var+'_low'], row['err_'+y_var+'_high']]]).T, color=rgba, marker=marker, ls='None', zorder=3, mec='black')
            except:
                pass
            try:
                ax.errorbar(row[x_var], row[y_var], yerr=row['err_'+y_var], color=rgba, marker=marker, ls='None', zorder=3, mec='black')
            except:
                pass
        else:
            ax.plot(row[x_var], row[y_var], color=rgba, marker=marker, ls='None', zorder=3, mec='black')
            
        ax.text(row[x_var], row[y_var], f"{int(row['groupID'])}", color='black')

    if add_leja:
        add_leja_sfms(ax)
        ax.legend()

    if plot_lims != 'None':
        ax.set_xlim(plot_lims[0], plot_lims[1])
        ax.set_ylim(plot_lims[2], plot_lims[3])


    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')

    if color_var!='None':
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(color_var, fontsize=full_page_axisfont)
        cbar.ax.tick_params(labelsize=full_page_axisfont)
    ax.tick_params(labelsize=full_page_axisfont)

    ax.set_xlabel(x_var, fontsize=full_page_axisfont)
    ax.set_ylabel(y_var, fontsize=full_page_axisfont)
    if y_var == 'override_flux_with_limit':
        ax.set_ylabel(x_var + '_older_method', fontsize=full_page_axisfont)
    if prospector_run_name != '':
        ax.set_ylabel('Prospector ' + y_var, fontsize=full_page_axisfont)

    if one_to_one:
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        ax.plot([-20, 20e60], [-20, 20e60], ls='--', color='red')
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        # ax.plot([0,1],[0,1], transform=ax.transAxes, ls='--', color='red')

    fig.savefig(imd.cluster_dir + f'/cluster_stats/{savename}.pdf', bbox_inches='tight')
    plt.close('all')

def plot_ratio(x_var_numerator, x_var_denominator, y_var_numerator, y_var_denominator, savename, color_var='None', plot_lims='None', one_to_one=False, ignore_groups=[], log=False, lower_limit=False):
    """Plots two columsn of cluster_summary_df against each other
    
    Parameters:
    savename (str): Name to save the figure under
    color_var (str): 'None' or the column to use as color
    plot_lims (list of 4): [xmin, xmax, ymin, ymax]
    one_to_one (boolean): Set to True to add a 1-1 line
    log (boolean): Set to True to make it a log-log plot
    lower_limit (boolean): Set to true to use lower limit SFRs and hollow out those points
    """

    fig, ax = plt.subplots(figsize = (8,8))

    if lower_limit == True:
        pass
        # y_var = y_var+'_with_limit'
        # savename = savename+'_with_limit'

    for i in range(len(cluster_summary_df)):
        if i in ignore_groups:
            continue
        row = cluster_summary_df.iloc[i]

        if color_var != 'None':
            cmap = mpl.cm.inferno
            norm = assign_color(color_var)
        else:
            rgba = 'black'

        # Make the point hollow if it's a lower limit
        if lower_limit == True:
            if row['flag_balmer_lower_limit']==1:
                marker='^'
            else:
                marker='o'
        else:
            marker='o'
        x_ratio = row[x_var_numerator] / row[x_var_denominator]
        y_ratio = row[y_var_numerator] / row[y_var_denominator]
        ax.plot(x_ratio, y_ratio, color=rgba, marker=marker, ls='None', zorder=3, mec='black')

    if plot_lims != 'None':
        ax.set_xlim(plot_lims[0], plot_lims[1])
        ax.set_ylim(plot_lims[2], plot_lims[3])

    if one_to_one:
        # ax.plot([-(1e18), 1e10], [-(1e18), 1e10], ls='--', color='red')
        ax.plot([0,1],[0,1], transform=ax.transAxes, ls='--', color='red')

    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_var, fontsize=full_page_axisfont)
    cbar.ax.tick_params(labelsize=full_page_axisfont)
    ax.tick_params(labelsize=full_page_axisfont)

    ax.set_xlabel(f'{x_var_numerator} / {x_var_denominator}', fontsize=full_page_axisfont)
    ax.set_ylabel(f'{y_var_numerator} / {y_var_denominator}', fontsize=full_page_axisfont)

    fig.savefig(imd.cluster_dir + f'/cluster_stats/{savename}.pdf', bbox_inches='tight')
    plt.close('all')

def plot_a_vs_b_paper(x_var, y_var, x_label, y_label, savename, axis_obj='False', color_var='None', plot_lims='None', lower_limit=False, one_to_one=False, ignore_groups=[], log=False, add_leja=False, yerr=False, prospector_run_name = '', fig='None', use_color_df=True, prospector_xerr=False, factor_of_2=False):
    """Plots two columsn of cluster_summary_df against each other
    
    Parameters:
    x_var (str): x variable to plot
    y_var (str): y variable to plot
    savename (str): Name to save the figure under
    color_var (str): 'None' or the column to use as color
    plot_lims (list of 4): [xmin, xmax, ymin, ymax]
    one_to_one (boolean): Set to True to add a 1-1 line
    log (boolean): Set to True to make it a log-log plot
    add_leja (boolean): If True, add the sfms from Leja 2022 to the plot
    yerr (boolean): Set to true to plot errorbars on the yaxis
    prospector_run_name (str): Set to the prospector run name if using
    """

    if axis_obj == 'False':
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        ax = axis_obj

    for i in range(len(cluster_summary_df)):
        if i in ignore_groups:
            continue
        row = cluster_summary_df.iloc[i]
        markersize = get_row_size(i)

        if prospector_run_name != '':
            if os.path.exists(imd.prospector_emission_fits_dir + f'/{prospector_run_name}_emission_fits/{i}_emission_fits.csv'):
                prospector_emission = ascii.read(imd.prospector_emission_fits_dir + f'/{prospector_run_name}_emission_fits/{i}_emission_fits.csv').to_pandas()
                ha_row = prospector_emission[prospector_emission['line_name']=='Halpha']
                row[y_var] = ha_row[y_var]
            else:
                continue

        if color_var != 'None':
            cmap = mpl.cm.inferno
            norm = assign_color(color_var)
            rgba = cmap(norm(row[color_var]))
        else:
            rgba = 'black'
        if use_color_df == True:
            rgba = get_row_color(i)
 
        # Make the point a triangle if it's a lower limit
        if lower_limit == True:
            if row['flag_balmer_lower_limit']==1:
                marker='^'
                yerr = False
            else:
                marker='o'
        else:
            marker='o'
        if yerr == True:
            if prospector_xerr == True:
                ax.errorbar(row[x_var], row[y_var], xerr=np.array([[row[x_var]-row[x_var.replace('_50','_16')], row[x_var.replace('_50','_84')]-row[x_var]]]).T, yerr=np.array([[row['err_'+y_var+'_low'], row['err_'+y_var+'_high']]]).T, color=rgba, marker=marker, ls='None', zorder=3, mec='black', ms=markersize)
            else:
                try:
                    ax.errorbar(row[x_var], row[y_var], yerr=np.array([[row['err_'+y_var+'_low'], row['err_'+y_var+'_high']]]).T, color=rgba, marker=marker, ls='None', zorder=3, mec='black', ms=markersize)
                except:
                    pass
                try:
                    ax.errorbar(row[x_var], row[y_var], yerr=np.array([[row[y_var]-row[y_var.replace('50','16')], row[y_var.replace('50','84')]-row[y_var]]]).T, color=rgba, marker=marker, ls='None', zorder=3, mec='black', ms=markersize)
                except:
                    pass
                try:
                    ax.errorbar(row[x_var], row[y_var], yerr=row['err_'+y_var], color=rgba, marker=marker, ls='None', zorder=3, mec='black', ms=markersize)
                except:
                    pass
        else:
            ax.plot(row[x_var], row[y_var], color=rgba, marker=marker, ls='None', zorder=3, mec='black', ms=markersize)
            
        # ax.text(row[x_var], row[y_var], f"{int(row['groupID'])}", color='black')

    if add_leja:
        redshift = 2
        mode = 'ridge'
        logmasses = np.arange(9, 11, 0.02)
        logSFRs = np.array([leja2022_sfms(logmass, redshift, mode) for logmass in logmasses])
        logssfrs = np.log10((10**logSFRs) / (10**logmasses))
        ax.plot(logmasses, logssfrs, color='black', marker='None', ls='-', zorder=1, label=f'Leja SFMS z={redshift}, type={mode}')
        ax.legend()

    if plot_lims != 'None':
        ax.set_xlim(plot_lims[0], plot_lims[1])
        ax.set_ylim(plot_lims[2], plot_lims[3])

    if log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    if color_var != 'None':
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(color_var, fontsize=full_page_axisfont)
        cbar.ax.tick_params(labelsize=full_page_axisfont)
    ax.tick_params(labelsize=full_page_axisfont)

    ax.set_xlabel(x_label, fontsize=full_page_axisfont)
    ax.set_ylabel(y_label, fontsize=full_page_axisfont)
    if prospector_run_name != '':
        ax.set_ylabel('Prospector ' + y_var, fontsize=full_page_axisfont)

    if one_to_one:
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        ax.plot([-20, 20e60], [-20, 20e60], ls='--', color='red')
        ax.plot([-10, 10], [-20, 20], ls='--', color='orange')
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        # ax.plot([0,1],[0,1], transform=ax.transAxes, ls='--', color='red')
    
    if axis_obj == 'False':
        fig.savefig(imd.cluster_dir + f'/paper_figures/{savename}.pdf', bbox_inches='tight')

def assign_color(color_var):
    if color_var=='balmer_dec':
        norm = mpl.colors.Normalize(vmin=3, vmax=5) 
    elif color_var=='balmer_dec_with_limit':
        norm = mpl.colors.Normalize(vmin=3, vmax=6) 
    elif color_var=='O3N2_metallicity':
        norm = mpl.colors.Normalize(vmin=8.2, vmax=9) 
    elif color_var=='norm_median_log_mass' or color_var=='median_log_mass':
        norm = mpl.colors.Normalize(vmin=9, vmax=11) 
    elif color_var=='prospector_log_mass':
        norm = mpl.colors.Normalize(vmin=12, vmax=14) 
    elif color_var=='median_U_V':
        norm = mpl.colors.Normalize(vmin=0.5, vmax=1.5) 
    else:
        norm = mpl.colors.Normalize(vmin=-10, vmax=10) 
    return norm

def add_leja_sfms(ax, mode='mean'):
    redshift = 2
    # mode = 'ridge'
    logmasses = np.arange(9, 11, 0.02)
    logSFRs = np.array([leja2022_sfms(logmass, redshift, mode) for logmass in logmasses])
    logssfrs = np.log10((10**logSFRs) / (10**logmasses))
    ax.plot(logmasses, logssfrs, color='black', marker='None', ls='--', zorder=1, label=f'Leja SFMS z={redshift}, {mode}')
 

def make_plots_a_vs_b(reduce_plot_count=False):
    """Plots variables in cluster_summary_df against each other
    
    Parameters:
    reduce_plot_count (boolean): True to make all plot, false to make only some
    """
    try:
        ignore_groups = imd.ignore_groups
    except:
        ignore_groups = []
    lower_limit = True

    if reduce_plot_count == False:
        # # SFR comparison plots
        plot_cluster_summaries('norm_median_halphas', 'ha_flux', 'sfrs/ha_flux_compare_norm', color_var='balmer_dec', plot_lims=[6e-18, 8e-16, 6e-18, 8e-16], one_to_one=True, ignore_groups=ignore_groups, log=True)
        plot_cluster_summaries('median_indiv_halphas', 'ha_flux', 'sfrs/ha_flux_compare_nonorm', color_var='median_log_mass', plot_lims=[6e-18, 8e-16, 6e-18, 8e-16], one_to_one=True, ignore_groups=ignore_groups, log=True)
        plot_cluster_summaries('median_halpha_luminosity', 'ha_flux', 'sfrs/ha_luminosity_compare_nonorm', color_var='median_log_mass', plot_lims=[3e41, 4e42, 3e41, 4e42], one_to_one=True, ignore_groups=ignore_groups, log=True, yerr=True)
        plot_cluster_summaries('median_indiv_halphas', 'ha_flux', 'sfrs/ha_flux_compare_nonorm_balmer', color_var='balmer_dec', plot_lims=[6e-18, 8e-16, 6e-18, 8e-16], one_to_one=True, ignore_groups=ignore_groups, log=True)
        plot_cluster_summaries('median_log_sfr', 'computed_log_sfr', 'sfrs/sfr_compare', color_var='balmer_dec', plot_lims=[0.3, 3, 0.3, 3], one_to_one=True, ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True)
        plot_cluster_summaries('median_log_ssfr', 'computed_log_ssfr', 'sfrs/ssfr_compare', color_var='balmer_dec', plot_lims=[-10.7, -6.5, -10.7, -6.5], one_to_one=True, ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True)
        plot_cluster_summaries('computed_log_sfr_with_limit', 'median_indiv_computed_log_sfr', 'sfrs/sfr_indiv_vs_cluster', color_var='balmer_dec', plot_lims=[0.3, 3, 0.3, 3], one_to_one=True, ignore_groups=ignore_groups, lower_limit=lower_limit)
        plot_cluster_summaries('computed_log_ssfr_with_limit', 'median_indiv_computed_log_ssfr', 'sfrs/ssfr_indiv_vs_cluster', color_var='balmer_dec', plot_lims=[-10.7, -6.5, -10.7, -6.5], one_to_one=True, ignore_groups=ignore_groups, lower_limit=lower_limit)
        plot_cluster_summaries('computed_log_sfr_with_limit', 'override_flux', 'sfrs/sfr_compare_new_vs_old_method', color_var='balmer_dec', plot_lims=[0.3, 3, 0.3, 3], one_to_one=True, ignore_groups=ignore_groups, lower_limit=lower_limit)
        plot_cluster_summaries('median_log_sfr', 'median_indiv_computed_log_sfr', 'sfrs/sfr_indiv_vs_mosdef', color_var='balmer_dec', plot_lims=[0.3, 3, 0.3, 3], one_to_one=True, ignore_groups=ignore_groups, lower_limit=lower_limit)

        # # Find which groups have accurate Ha and Hb measurements:
        ignore_groups = np.array(cluster_summary_df[cluster_summary_df['err_balmer_dec_high']>1].index)
        plot_cluster_summaries('norm_median_halphas', 'ha_flux', 'sfrs/ha_flux_compare_balmer_accurate', color_var='balmer_dec', plot_lims=[6e-18, 8e-16, 6e-18, 8e-16], one_to_one=True, ignore_groups=ignore_groups, log=True)
        plot_cluster_summaries('median_log_sfr', 'computed_log_sfr', 'sfrs/sfr_compare_balmer_accurate', color_var='balmer_dec', plot_lims=[0.3, 3, 0.3, 3], one_to_one=True, ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True)
        plot_cluster_summaries('median_log_ssfr', 'computed_log_ssfr', 'sfrs/ssfr_compare_balmer_accurate', color_var='balmer_dec', plot_lims=[-10.7, -6.5, -10.7, -6.5], one_to_one=True, ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True)
        try:
            ignore_groups = imd.ignore_groups
        except:
            ignore_groups = []

        # # SFMS
        plot_cluster_summaries('median_log_mass', 'median_log_ssfr', 'sfrs/sfms', color_var='O3N2_metallicity', ignore_groups=ignore_groups)
        plot_cluster_summaries('median_log_mass', 'computed_log_ssfr', 'sfrs/sfms_computed', color_var='O3N2_metallicity', ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True, plot_lims=[9, 11, -10.5, -7.5])
        plot_cluster_summaries('median_log_mass', 'computed_log_ssfr', 'sfrs/sfms_computed_balmercolor', color_var='balmer_dec', ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True, plot_lims=[9, 11, -10.5, -7.5])
        plot_cluster_summaries('median_log_mass', 'computed_log_ssfr', 'sfrs/sfms_with_Leja', color_var='balmer_dec', ignore_groups=ignore_groups, lower_limit=lower_limit, add_leja=True, yerr=True, plot_lims=[9, 11, -10.5, -7.5])

        # # SFR Mass
        plot_cluster_summaries('median_log_mass', 'computed_log_sfr', 'sfrs/sfr_mass_lower_limit', color_var='balmer_dec', ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True, plot_lims=[9, 11, 0.3, 3])
        plot_cluster_summaries('median_log_mass', 'median_log_sfr', 'sfrs/sfr_mass_median', color_var='balmer_dec', ignore_groups=ignore_groups, plot_lims=[9, 11, 0.3, 3])

        # # Halpha compare
        plot_cluster_summaries('ha_flux', 'median_indiv_halphas', 'sfrs/halpha_norm_compare', color_var='balmer_dec', ignore_groups=ignore_groups, one_to_one=True, log=True)

        # #AV comparison
        plot_cluster_summaries('AV', 'balmer_av', 'sfrs/av_compare', color_var='norm_median_log_mass', ignore_groups=ignore_groups, one_to_one=True, plot_lims=[0, 4.5, 0, 4.5], lower_limit=lower_limit, yerr=True)

        # # Trying to diagnose what makes the SFR high
        imd.check_and_make_dir(imd.cluster_dir + f'/cluster_stats/sfrs/diagnostics/')
        plot_cluster_summaries('median_indiv_balmer_decs', 'balmer_dec', 'sfrs/diagnostics/balmer_dec_compare', color_var='median_log_mass', ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True, one_to_one=True)
        sys.exit()
        plot_cluster_summaries('redshift', 'computed_log_sfr', 'sfrs/diagnostics/sfr_z_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True)
        plot_cluster_summaries('target_galaxy_redshifts', 'computed_log_sfr', 'sfrs/diagnostics/sfr_ztarget_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True)
        plot_cluster_summaries('target_galaxy_median_log_mass', 'computed_log_sfr', 'sfrs/diagnostics/sfr_masstarget_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True)
        plot_cluster_summaries('median_log_mass', 'computed_log_sfr', 'sfrs/diagnostics/sfr_massgroup_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True)
        plot_cluster_summaries('ha_flux', 'computed_log_sfr', 'sfrs/diagnostics/sfr_haflux_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True)
        plot_cluster_summaries('balmer_dec', 'computed_log_sfr', 'sfrs/diagnostics/sfr_balmer_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True)
        plot_cluster_summaries('balmer_av', 'computed_log_sfr', 'sfrs/diagnostics/sfr_balmerav_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True)
        plot_cluster_summaries('balmer_dec_with_limit', 'computed_log_sfr', 'sfrs/diagnostics/sfr_balmerlimit_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True)
        plot_cluster_summaries('ha_flux', 'computed_log_sfr', 'sfrs/diagnostics/sfr_haflux_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True)
        plot_cluster_summaries('hb_flux', 'computed_log_sfr', 'sfrs/diagnostics/sfr_hbflux_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit, yerr=True)
        
    # #SNR Plots
    # plot_cluster_summaries('hb_snr', 'balmer_dec_snr', 'sfrs/hbeta_balmer_snr', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, one_to_one=True)

    # Prospector emission - to use, set the yvar to the prospector name, and prospector_run_name to be accurate
    imd.check_and_make_dir(imd.cluster_dir + '/cluster_stats/prospector/')
    plot_cluster_summaries('ha_flux', 'luminosity', 'sfrs/prospector_ha_compare', color_var='median_log_mass', one_to_one=True, ignore_groups=ignore_groups, log=True, prospector_run_name='dust_type4')

    # Prospector eline properties:
    plot_cluster_summaries('prospector_balmer_dec', 'balmer_dec', 'prospector/balmer_dec_compare', color_var='median_log_mass', ignore_groups=ignore_groups, lower_limit=lower_limit, one_to_one=True, plot_lims=[2.7, 6.5, 2.7, 6.5])
    plot_cluster_summaries('O3N2_metallicity', 'prospector_O3N2_metallicity', 'prospector/metallicity_compare', color_var='median_log_mass', ignore_groups=ignore_groups, one_to_one=True, plot_lims=[7.7, 9, 7.7, 9])
    plot_cluster_summaries('O3N2_metallicity', 'prospector_O3N2_metallicity', 'prospector/metallicity_compare_prospmassscolor', color_var='prospector_log_mass', ignore_groups=ignore_groups, one_to_one=True, plot_lims=[7.7, 9, 7.7, 9])
    plot_cluster_summaries('O3N2_metallicity', 'prospector_O3N2_metallicity', 'prospector/metallicity_compare_balmercolor', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, one_to_one=True, plot_lims=[7.7, 9, 7.7, 9])
    plot_cluster_summaries('cluster_av_prospector_log_ssfr', 'prospector_log_ssfr', 'prospector/prospector_ssfr_compare', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, one_to_one=True, plot_lims=[-11, -7.5, -11, -7.5])
    plot_cluster_summaries('cluster_av_prospector_log_ssfr', 'computed_log_ssfr', 'prospector/ssfr_compare_to_cluster', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit, one_to_one=True, plot_lims=[-11, -7.5, -11, -7.5])

    # Prospector Dust Index
    plot_cluster_summaries('computed_log_ssfr_with_limit', 'dustindex50', 'prospector/dust_index_ssfr', color_var='balmer_dec', ignore_groups=ignore_groups)
    plot_cluster_summaries('median_log_mass', 'dustindex50', 'prospector/dust_index_mass', color_var='balmer_dec', ignore_groups=ignore_groups)
    plot_cluster_summaries('balmer_dec', 'dustindex50', 'prospector/dust_index_balmer', color_var='balmer_dec', ignore_groups=ignore_groups)
    plot_cluster_summaries('AV', 'dustindex50', 'prospector/dust_index_av', color_var='balmer_dec', ignore_groups=ignore_groups)
    plot_cluster_summaries('logzsol50', 'dustindex50', 'prospector/dust_index_prospmetals', color_var='balmer_dec', ignore_groups=ignore_groups)
    plot_cluster_summaries('O3N2_metallicity', 'dustindex50', 'prospector/dust_index_metals', color_var='balmer_dec', ignore_groups=ignore_groups)

    # Prospector AV
    plot_cluster_summaries('median_log_mass', 'dust2_50', 'prospector/dust2_mass', color_var='balmer_dec', ignore_groups=ignore_groups)
    plot_cluster_summaries('AV', 'dust2_50', 'prospector/dust2_medianAVA', color_var='balmer_dec', ignore_groups=ignore_groups, one_to_one=True)

# make_plots_a_vs_b(reduce_plot_count=True)