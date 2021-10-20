# Plot a histogram of the axis ratios of the galaxies being used
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.patheffects as path_effects


def plot_ar_hist():

    ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()
    
    fig, ax = plt.subplots(figsize=(8,8))

    ax.hist(ar_df['use_ratio'], 20, color='black')

    ax.axvline(0.4, ls='--', color='red')
    ax.axvline(0.7, ls='--', color='red')

    ax.set_xlabel('Axis Ratio', fontsize=14) 
    ax.set_ylabel('Number of Galaxies', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.text(-0.07, -10, 'Edge-on', fontsize=14, zorder=100)
    ax.text(0.95, -10, 'Face-on', fontsize=14, zorder=100)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, 110)
    fig.savefig(imd.axis_output_dir + '/ar_histogram.pdf')

plot_ar_hist()