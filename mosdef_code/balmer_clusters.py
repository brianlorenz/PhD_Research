# Plots the Balmer dec of the clusters

import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
from astropy.io import ascii
import numpy as np
import matplotlib as mpl
from plot_vals import *

def plot_balmer_dec_clusters():
    clusters_summary_df = ascii.read(imd.loc_cluster_summary_df).to_pandas()

    fig, ax = plt.subplots(figsize = (9,8))
    cmap = mpl.cm.inferno
    norm = mpl.colors.Normalize(vmin=-9.6, vmax=-8.2) 

    for i in range(len(clusters_summary_df)):
        row = clusters_summary_df.iloc[i]
        if row['balmer_dec'] < -90:
            continue

        
        rgba = cmap(norm(row['log_ssfr']))

        ax.plot(row['log_mass'], row['balmer_dec'], marker='o', ls='None', color=rgba)
        ax.text(row['log_mass']+0.02, row['balmer_dec']+0.02, f'{int(row["groupID"])}')
    ax.set_xlabel(stellar_mass_label, fontsize=full_page_axisfont)
    ax.set_ylabel('Balmer Decrement', fontsize=full_page_axisfont)
    ax.tick_params(labelsize=full_page_axisfont-2)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('log_ssfr', fontsize=full_page_axisfont)
    cbar.ax.tick_params(labelsize=full_page_axisfont-2)
    fig.savefig(imd.cluster_dir+'/cluster_stats/balmer_mass.pdf')

plot_balmer_dec_clusters()