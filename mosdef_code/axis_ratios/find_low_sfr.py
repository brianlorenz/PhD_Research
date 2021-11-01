# Finds the galaxies that have SFR lower limits and plots them in sfr/mass space

import numpy as np
import pandas as pd
from astropy.io import ascii, fits
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.patheffects as path_effects
from astropy.table import Table






def plot_low_sfrs():
    
    
    ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()

    # get the number with bad ha
    bad_ha_idx = ar_df['ha_detflag_sfr'] == 1
    nondet_ha_idx = ar_df['ha_detflag_sfr'] == -999.0
    
    n_galaxies = len(ar_df)
    n_bad_ha = len(ar_df[bad_ha_idx])
    n_nondet_ha = len(ar_df[nondet_ha_idx])

    ar_df = ar_df[np.logical_and(~bad_ha_idx, ~nondet_ha_idx)]

    # get the number filtered out by hbeta being undetected
    sfr_cut_idx = ar_df['hb_detflag_sfr'] == -999.0

    n_good_ha = len(ar_df)
    n_dropped = len(ar_df[sfr_cut_idx])

    ar_df = ar_df[~sfr_cut_idx]

    ar_df['log_ssfr'] = np.log10(ar_df['sfr']/10**ar_df['log_mass'])

    # Find those with non-detection s in hbeta (lower limits)
    low_lim_idx = ar_df['hb_detflag_sfr'] == 1.0

    # Get the number of remaining galaxies in number of lower limits
    n_lower_lim = len(ar_df[low_lim_idx])
    n_remaining = len(ar_df[~low_lim_idx])

    fig, ax = plt.subplots(figsize=(8,8))

    # Plot good ones in green, lower limits in red
    ax.plot(ar_df[~low_lim_idx]['log_mass'], ar_df[~low_lim_idx]['log_ssfr'], color='green', ls='None', marker='o')
    ax.plot(ar_df[low_lim_idx]['log_mass'], ar_df[low_lim_idx]['log_ssfr'], color='red', ls='None', marker='o')


    text_labels = [
        f'Total Number: {n_galaxies}',
        f'Halpha nondet: {n_nondet_ha}',
        f'Halpha upper_lim: {n_bad_ha}',
        f'Hbeta nondet (drop): {n_dropped}',
        f'Hbeta upper_lim (keep): {n_lower_lim}',
        f'Good both lines: {n_remaining}',
    ]

    for k in range(len(text_labels)):
            ax.text(9.1, -6.2-(0.2*k), text_labels[k], fontsize=12, color='black')

    ax.set_xlabel('log(Stellar Mass)', fontsize=14)
    ax.set_ylabel('log(ssfr)', fontsize=14)
    ax.set_xlim(9, 11)
    ax.set_ylim(-11, -6)
    ax.tick_params(labelsize=12)

    fig.savefig(imd.axis_output_dir + '/dropped_sfrs.pdf')


plot_low_sfrs()