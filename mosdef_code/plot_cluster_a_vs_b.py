import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot_vals import *
from leja_sfms_redshift import leja2022_sfms
cluster_summary_df = imd.read_cluster_summary_df()

def plot_cluster_summaries(x_var, y_var, savename, color_var='None', plot_lims='None', one_to_one=False, ignore_groups=[], log=False, lower_limit=False, add_leja_sfms=False):
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
    add_leja_sfms (boolean): If True, add the sfms from Leja 2022 to the plot
    """

    fig, ax = plt.subplots(figsize = (8,8))

    if lower_limit == True:
        y_var = y_var+'_with_limit'
        savename = savename+'_with_limit'

    for i in range(len(cluster_summary_df)):
        if i in ignore_groups:
            continue
        row = cluster_summary_df.iloc[i]

        if color_var != 'None':
            cmap = mpl.cm.inferno
            if color_var=='balmer_dec':
                norm = mpl.colors.Normalize(vmin=3, vmax=5) 
            elif color_var=='balmer_dec_with_limit':
                norm = mpl.colors.Normalize(vmin=3, vmax=6) 
            elif color_var=='O3N2_metallicity':
                norm = mpl.colors.Normalize(vmin=8.2, vmax=9) 
            elif color_var=='norm_median_log_mass':
                norm = mpl.colors.Normalize(vmin=9, vmax=11) 
            else:
                norm = mpl.colors.Normalize(vmin=-10, vmax=10) 
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
        ax.plot(row[x_var], row[y_var], color=rgba, marker=marker, ls='None', zorder=3, mec='black')
        ax.text(row[x_var], row[y_var], f"{int(row['groupID'])}", color='black')

    if add_leja_sfms:
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

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_var, fontsize=full_page_axisfont)
    cbar.ax.tick_params(labelsize=full_page_axisfont)
    ax.tick_params(labelsize=full_page_axisfont)

    ax.set_xlabel(x_var, fontsize=full_page_axisfont)
    ax.set_ylabel(y_var, fontsize=full_page_axisfont)

    if one_to_one:
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        ax.plot([-20, 20], [-20, 20], ls='--', color='red')
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        # ax.plot([0,1],[0,1], transform=ax.transAxes, ls='--', color='red')

    fig.savefig(imd.cluster_dir + f'/cluster_stats/{savename}.pdf', bbox_inches='tight')

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
            if color_var=='balmer_dec':
                norm = mpl.colors.Normalize(vmin=3, vmax=5) 
            elif color_var=='balmer_dec_with_limit':
                norm = mpl.colors.Normalize(vmin=3, vmax=6) 
            elif color_var=='O3N2_metallicity':
                norm = mpl.colors.Normalize(vmin=8.2, vmax=9) 
            elif color_var=='norm_median_log_mass':
                norm = mpl.colors.Normalize(vmin=9, vmax=11) 
            else:
                norm = mpl.colors.Normalize(vmin=-10, vmax=10) 
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

def make_plots_a_vs_b():
    ignore_groups = imd.ignore_groups
    lower_limit = True

    # SFR comparison plots
    plot_cluster_summaries('norm_median_halphas', 'ha_flux', 'sfrs/ha_flux_compare', color_var='balmer_dec', plot_lims=[6e-18, 8e-16, 6e-18, 8e-16], one_to_one=True, ignore_groups=ignore_groups, log=True)
    plot_cluster_summaries('median_log_sfr', 'computed_log_sfr', 'sfrs/sfr_compare', color_var='balmer_dec', plot_lims=[0.3, 3, 0.3, 3], one_to_one=True, ignore_groups=ignore_groups, lower_limit=lower_limit)
    plot_cluster_summaries('median_log_ssfr', 'computed_log_ssfr', 'sfrs/ssfr_compare', color_var='balmer_dec', plot_lims=[-10.7, -6.5, -10.7, -6.5], one_to_one=True, ignore_groups=ignore_groups, lower_limit=lower_limit)

    # Find which groups have accurate Ha and Hb measurements:
    ignore_groups = np.array(cluster_summary_df[cluster_summary_df['err_balmer_dec_high']>1].index)
    plot_cluster_summaries('norm_median_halphas', 'ha_flux', 'sfrs/ha_flux_compare_balmer_accurate', color_var='balmer_dec', plot_lims=[6e-18, 8e-16, 6e-18, 8e-16], one_to_one=True, ignore_groups=ignore_groups, log=True)
    plot_cluster_summaries('median_log_sfr', 'computed_log_sfr', 'sfrs/sfr_compare_balmer_accurate', color_var='balmer_dec', plot_lims=[0.3, 3, 0.3, 3], one_to_one=True, ignore_groups=ignore_groups, lower_limit=lower_limit)
    plot_cluster_summaries('median_log_ssfr', 'computed_log_ssfr', 'sfrs/ssfr_compare_balmer_accurate', color_var='balmer_dec', plot_lims=[-10.7, -6.5, -10.7, -6.5], one_to_one=True, ignore_groups=ignore_groups, lower_limit=lower_limit)
    ignore_groups = imd.ignore_groups

    # SFMS
    plot_cluster_summaries('median_log_mass', 'median_log_ssfr', 'sfrs/sfms', color_var='O3N2_metallicity', ignore_groups=ignore_groups)
    plot_cluster_summaries('median_log_mass', 'computed_log_ssfr', 'sfrs/sfms_computed', color_var='O3N2_metallicity', ignore_groups=ignore_groups, lower_limit=lower_limit)
    plot_cluster_summaries('median_log_mass', 'computed_log_ssfr', 'sfrs/sfms_computed_balmercolor', color_var='balmer_dec', ignore_groups=ignore_groups, lower_limit=lower_limit)
    plot_cluster_summaries('median_log_mass', 'computed_log_ssfr', 'sfrs/sfms_with_Leja', color_var='balmer_dec', ignore_groups=ignore_groups, lower_limit=lower_limit, add_leja_sfms=True)

    # SFR Mass
    plot_cluster_summaries('median_log_mass', 'computed_log_sfr', 'sfrs/sfr_mass_lower_limit', color_var='balmer_dec', ignore_groups=ignore_groups, lower_limit=lower_limit)
    plot_cluster_summaries('median_log_mass', 'median_log_sfr', 'sfrs/sfr_mass_median', color_var='balmer_dec', ignore_groups=ignore_groups)

    # Halpha compare
    plot_cluster_summaries('ha_flux', 'median_indiv_halphas', 'sfrs/halpha_norm_compare', color_var='balmer_dec', ignore_groups=ignore_groups, one_to_one=True, log=True)

    #AV comparison
    plot_cluster_summaries('AV', 'balmer_av', 'sfrs/av_compare', color_var='norm_median_log_mass', ignore_groups=ignore_groups, one_to_one=True, plot_lims=[0, 4.5, 0, 4.5], lower_limit=lower_limit)
    plot_cluster_summaries('AV', 'balmer_av', 'sfrs/av_compare', color_var='norm_median_log_mass', ignore_groups=ignore_groups, one_to_one=True, plot_lims=[0, 4.5, 0, 4.5], lower_limit=lower_limit)

    # Trying to diagnose what makes the SFR high
    plot_cluster_summaries('redshift', 'computed_log_sfr', 'sfrs/diagnostics/sfr_z_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit)
    plot_cluster_summaries('target_galaxy_redshifts', 'computed_log_sfr', 'sfrs/diagnostics/sfr_ztarget_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit)
    plot_cluster_summaries('target_galaxy_median_log_mass', 'computed_log_sfr', 'sfrs/diagnostics/sfr_masstarget_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit)
    plot_cluster_summaries('median_log_mass', 'computed_log_sfr', 'sfrs/diagnostics/sfr_massgroup_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit)
    plot_cluster_summaries('ha_flux', 'computed_log_sfr', 'sfrs/diagnostics/sfr_haflux_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit)
    plot_cluster_summaries('balmer_dec', 'computed_log_sfr', 'sfrs/diagnostics/sfr_balmer_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit)
    plot_cluster_summaries('balmer_av', 'computed_log_sfr', 'sfrs/diagnostics/sfr_balmerav_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit)
    plot_cluster_summaries('balmer_dec_with_limit', 'computed_log_sfr', 'sfrs/diagnostics/sfr_balmerlimit_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit)
    plot_cluster_summaries('ha_flux', 'computed_log_sfr', 'sfrs/diagnostics/sfr_haflux_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit)
    plot_cluster_summaries('hb_flux', 'computed_log_sfr', 'sfrs/diagnostics/sfr_hbflux_lower_limit', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, lower_limit=lower_limit)
    
    #SNR Plots
    plot_cluster_summaries('hb_snr', 'balmer_dec_snr', 'sfrs/hbeta_balmer_snr', color_var='balmer_dec_with_limit', ignore_groups=ignore_groups, one_to_one=True)
# make_plots_a_vs_b()