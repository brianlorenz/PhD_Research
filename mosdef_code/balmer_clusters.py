# Plots the Balmer dec of the clusters

import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
from astropy.io import ascii
import numpy as np
import matplotlib as mpl
from plot_vals import *

def plot_balmer_dec_clusters(plot_var='balmer_dec', errorbar=True, clean=False):
    """
    
    Parameters:
    clean (boolean): Set to true to remove point labels and make a cleaner plot

    """
    
    clusters_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()

    if plot_var=='composite_beta':
        composite_beta_df = ascii.read(imd.loc_composite_beta_df).to_pandas()
        clusters_summary_df = clusters_summary_df.merge(composite_beta_df, on='groupID')
        errorbar = False

    fig, ax = plt.subplots(figsize = (9,8))
    cmap = mpl.cm.inferno
    norm = mpl.colors.Normalize(vmin=-9.6, vmax=-8.2) 

    for i in range(len(clusters_summary_df)):
        row = clusters_summary_df.iloc[i]
        if plot_var == 'balmer_dec':
            if row['balmer_dec'] < -90:
                continue
    

        
        rgba = cmap(norm(row['median_log_ssfr']))
        
        if errorbar == True:
            ax.errorbar(row['median_log_mass'], row[plot_var], yerr=np.array([[row[f'err_{plot_var}_low'], row[f'err_{plot_var}_high']]]).T, marker='o', ls='None', color=rgba)
        else:
            ax.plot(row['median_log_mass'], row[plot_var], marker='o', ls='None', color=rgba)
        if clean==False:
            ax.text(row['median_log_mass']+0.02, row[plot_var]+0.02, f'{int(row["groupID"])}')
    ax.set_xlabel(stellar_mass_label, fontsize=full_page_axisfont)
    ax.set_ylabel(plot_var, fontsize=full_page_axisfont)
    if plot_var=='balmer_dec':
        ax.set_ylim(2, 11)
        ax.set_ylabel('Balmer Dec', fontsize=full_page_axisfont)
    if plot_var=='AV':
        pass
        #ax.set_ylim(2, 11)
    if plot_var=='beta':
        pass
        #ax.set_ylim(2, 11)
    if plot_var=='O3N2_metallicity':
        ax.set_ylim(8, 9.5)
    ax.tick_params(labelsize=full_page_axisfont-2)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('log(ssfr)', fontsize=full_page_axisfont)
    cbar.ax.tick_params(labelsize=full_page_axisfont-2)
    if clean==True:
        add_str='_nolabel'
    else:
        add_str = ''
    fig.savefig(imd.cluster_dir+f'/cluster_stats/{plot_var}_mass{add_str}.pdf', bbox_inches='tight')

plot_balmer_dec_clusters(clean=True)
# plot_balmer_dec_clusters(plot_var='AV')
# plot_balmer_dec_clusters(plot_var='beta')
# plot_balmer_dec_clusters(plot_var='O3N2_metallicity')
# plot_balmer_dec_clusters(plot_var='composite_beta')