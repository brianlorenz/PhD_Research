import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot_vals import *
cluster_summary_df = imd.read_cluster_summary_df()

def plot_cluster_summaries(x_var, y_var, savename, color_var='None', plot_lims='None', one_to_one=False, ignore_groups=[], log=False):
    """Plots two columsn of cluster_summary_df against each other
    
    Parameters:
    x_var (str): x variable to plot
    y_var (str): y variable to plot
    savename (str): Name to save the figure under
    color_var (str): 'None' or the column to use as color
    plot_lims (list of 4): [xmin, xmax, ymin, ymax]
    one_to_one (boolean): Set to True to add a 1-1 line
    log (boolean): Set to True to make it a log-log plot
    """

    fig, ax = plt.subplots(figsize = (8,8))

    for i in range(len(cluster_summary_df)):
        if i in ignore_groups:
            continue
        row = cluster_summary_df.iloc[i]

        if color_var != 'None':
            cmap = mpl.cm.inferno
            if color_var=='balmer_dec':
                norm = mpl.colors.Normalize(vmin=3, vmax=10) 
            elif color_var=='O3N2_metallicity':
                norm = mpl.colors.Normalize(vmin=8.2, vmax=9) 
            elif color_var=='norm_median_log_mass':
                norm = mpl.colors.Normalize(vmin=9, vmax=11) 
            else:
                norm = mpl.colors.Normalize(vmin=-10, vmax=10) 
            rgba = cmap(norm(row[color_var]))
        else:
            rgba = 'black'

        # yerr=np.array([[row['err_O3N2_metallicity_low'], row['err_O3N2_metallicity_high']]]).T
        ax.plot(row[x_var], row[y_var], color=rgba, marker='o', ls='None', zorder=3, mec='black')

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

    ax.set_xlabel(x_var, fontsize=full_page_axisfont)
    ax.set_ylabel(y_var, fontsize=full_page_axisfont)

    fig.savefig(imd.cluster_dir + f'/cluster_stats/{savename}.pdf', bbox_inches='tight')

ignore_groups = imd.ignore_groups

# SFR comparison plots
plot_cluster_summaries('norm_median_halphas', 'ha_flux', 'sfrs/ha_flux_compare', color_var='balmer_dec', plot_lims=[6e-18, 8e-16, 6e-18, 8e-16], one_to_one=True, ignore_groups=ignore_groups, log=True)
plot_cluster_summaries('median_log_sfr', 'computed_log_sfr', 'sfrs/sfr_compare', color_var='balmer_dec', plot_lims=[0.3, 3.5, 0.3, 3.5], one_to_one=True, ignore_groups=ignore_groups)
plot_cluster_summaries('median_log_ssfr', 'computed_log_ssfr', 'sfrs/ssfr_compare', color_var='balmer_dec', plot_lims=[-10.7, -6.5, -10.7, -6.5], one_to_one=True, ignore_groups=ignore_groups)

# Find which groups have accurate Ha and Hb measurements:
ignore_groups = np.array(cluster_summary_df[cluster_summary_df['err_balmer_dec_high']>1].index)
plot_cluster_summaries('norm_median_halphas', 'ha_flux', 'sfrs/ha_flux_compare_balmer_accurate', color_var='balmer_dec', plot_lims=[6e-18, 8e-16, 6e-18, 8e-16], one_to_one=True, ignore_groups=ignore_groups, log=True)
plot_cluster_summaries('median_log_sfr', 'computed_log_sfr', 'sfrs/sfr_compare_balmer_accurate', color_var='balmer_dec', plot_lims=[0.3, 3.5, 0.3, 3.5], one_to_one=True, ignore_groups=ignore_groups)
plot_cluster_summaries('median_log_ssfr', 'computed_log_ssfr', 'sfrs/ssfr_compare_balmer_accurate', color_var='balmer_dec', plot_lims=[-10.7, -6.5, -10.7, -6.5], one_to_one=True, ignore_groups=ignore_groups)
ignore_groups = imd.ignore_groups

# SFMS
plot_cluster_summaries('median_log_mass', 'median_log_ssfr', 'sfrs/sfms', color_var='O3N2_metallicity', ignore_groups=ignore_groups)
plot_cluster_summaries('median_log_mass', 'computed_log_ssfr', 'sfrs/sfms_computed', color_var='O3N2_metallicity', ignore_groups=ignore_groups)
plot_cluster_summaries('median_log_mass', 'computed_log_ssfr', 'sfrs/sfms_computed_balmercolor', color_var='balmer_dec', ignore_groups=ignore_groups)

#AV comparison
plot_cluster_summaries('AV', 'balmer_av', 'sfrs/av_compare', color_var='norm_median_log_mass', ignore_groups=ignore_groups, one_to_one=True, plot_lims=[0, 4.5, 0, 4.5])