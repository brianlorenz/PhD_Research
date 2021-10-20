# Plot the sfr measures against each other to see how comparable they are
import numpy as np
import pandas as pd
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.patheffects as path_effects
from astropy.table import Table


def plot_sfr2_vs_sfrcor():
    dat = Table.read(imd.loc_sfrs_latest, format='fits')
    sfrs_df = dat.to_pandas()

    sfrs_df = sfrs_df[sfrs_df['SFR2'] > -10]
    sfrs_df = sfrs_df[sfrs_df['SFR_CORR'] > -10]

    fig, ax = plt.subplots(figsize=(8,8))
    ax.scatter(sfrs_df['SFR2'], sfrs_df['SFR_CORR'], color='black')
    ax.set_xlabel('SFR2', fontsize=14)
    ax.set_ylabel('SFR_CORR', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.tick_params(labelsize=12)

    fig.savefig(imd.axis_output_dir + '/sfr2_vs_sfrcorr.pdf')

plot_sfr2_vs_sfrcor()