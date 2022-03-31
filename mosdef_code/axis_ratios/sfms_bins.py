
import os
import numpy as np
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd


def plot_sfms_bins(save_name, nbins, split_by):
    '''Divide the galaxies in sfr_mass space along the sfms and plot it'''
    
    group_dfs = [ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_group_dfs/{i}.csv').to_pandas() for i in range(nbins)]
    colors = ['red', 'orange', 'blue', 'black', 'brown', 'green', 'pink', 'grey', 'purple', 'cyan', 'navy', 'magenta']

    fig, ax = plt.subplots(figsize=(8,8))

    for i in range(len(group_dfs)):
        df = group_dfs[i]
        ax.plot(df['log_mass'], df[split_by], color=colors[i], ls='None', marker='o')
    x = np.linspace(-5, 5, 2)
    y = 2*x+1
    plt.plot(x, y, '-r', label='y=2x+1')

    ax.set_xlim(8.95, 11.05)
    
    ax.set_xlabel('log(Stellar Mass)')
    ax.set_ylabel('log(SFR)')
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/sfr_mass.pdf')

plot_sfms_bins('both_sfms_6bin_median_2axis', 12, 'log_use_sfr')
#low cut - (9.5, 0.3), (11.0, 1.9)  y = 1.07x-9.83
#high cut - (9.0, 0.5), (10.5, 2.0) y = 1.07x-8.6


