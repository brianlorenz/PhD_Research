import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
import matplotlib as mpl
from plot_vals import *


ignore_groups = imd.ignore_groups
cluster_summary_df = imd.read_cluster_summary_df()


def make_vis_plot(x_var, y_var, x_var_label, y_var_label, savename, color_var_name='None', xlim='None', ylim='None'):
    fig, ax = plt.subplots(figsize = (8,8))

    for i in range(len(cluster_summary_df)):
        if i in ignore_groups:
            continue
        row = cluster_summary_df.iloc[i]

        if color_var_name != 'None':
            cmap = mpl.cm.inferno
            if color_var_name=='balmer_dec':
                norm = mpl.colors.Normalize(vmin=3, vmax=5) 
            elif color_var_name=='balmer_dec_with_limit':
                norm = mpl.colors.Normalize(vmin=3, vmax=6) 
            elif color_var_name=='O3N2_metallicity':
                norm = mpl.colors.Normalize(vmin=8.2, vmax=9) 
            elif color_var_name=='norm_median_log_mass' or color_var_name=='median_log_mass':
                norm = mpl.colors.Normalize(vmin=9, vmax=11) 
            else:
                norm = mpl.colors.Normalize(vmin=-10, vmax=10) 
            rgba = cmap(norm(row[color_var_name]))
        else:
            rgba = 'black'

        # Make the point hollow if it's a lower limit
        if row['flag_balmer_lower_limit']==1:
            marker='^'
        else:
            marker='o'

        # ax.errorbar(row[x_var], row[y_var], yerr=np.array([[row['err_'+y_var+'_low'], row['err_'+y_var+'_high']]]).T, color=rgba, marker=marker, ls='None', zorder=3, mec='black')
        ax.plot(x_var[i], y_var[i], color=rgba, marker=marker, ls='None', zorder=3, mec='black')

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_var_name, fontsize=full_page_axisfont)
    cbar.ax.tick_params(labelsize=full_page_axisfont)
    ax.tick_params(labelsize=full_page_axisfont)

    ax.set_xlabel(x_var_label, fontsize=full_page_axisfont)
    ax.set_ylabel(y_var_label, fontsize=full_page_axisfont)

    if xlim!='None':
        ax.set_xlim(xlim)
    if ylim!='None':
        ax.set_ylim(ylim)

    imd.check_and_make_dir(imd.cluster_dir + f'/cluster_stats/dust_model_vis')
    fig.savefig(imd.cluster_dir + f'/cluster_stats/dust_model_vis/{savename}.pdf', bbox_inches='tight')


logsfrs = cluster_summary_df['computed_log_sfr_with_limit']
metallicities = cluster_summary_df['O3N2_metallicity']
res = cluster_summary_df['median_re']
balmer_decs = cluster_summary_df['balmer_dec']
masses = cluster_summary_df['median_log_mass']

logsfr_times_metals = logsfrs*metallicities
make_vis_plot(logsfr_times_metals, balmer_decs,'log(SFR)*Metallicity', 'Balmer dec', 'sfr_times_metallicity', color_var_name='median_log_mass', ylim=[0,10])
logsfr_times_metals_withre = np.log10(((10**logsfrs)/(res**2))**(1/1.4))*metallicities
# make_vis_plot(logsfr_times_metals_withre, balmer_decs,'Formula from paper', 'Balmer dec', 'sfr_times_metallicity_with_re', color_var_name='median_log_mass', ylim=[0,10])


make_vis_plot(logsfr_times_metals, masses,'log(SFR)*Metallicity', 'Mass', 'sfr_times_metallicity_vs_mass', color_var_name='balmer_dec')
