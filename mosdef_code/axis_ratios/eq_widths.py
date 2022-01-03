# Equivalent width vs mass figures
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.patheffects as path_effects
from axis_ratio_funcs import read_interp_axis_ratio, filter_ar_df



def plot_mass_vs_eqha():
    ar_df = read_interp_axis_ratio()

    ar_df = filter_ar_df(ar_df)

    fig, ax = plt.subplots(figsize=(8,8))


    ar_df = ar_df[ar_df['eq_width_ha'] > 0]
    ax.errorbar(ar_df['log_mass'], ar_df['eq_width_ha'], yerr=ar_df['err_eq_width_ha'], marker='o', ls='None', color='black')

    # Set the axis limits
    xlims = (9.0, 11.0)
    ylims = (0, 2000)
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)

    ax.set_xlabel('log(Stellar Mass)', fontsize=14) 
    ax.set_ylabel('Halpha eq width', fontsize=14)

    ax.tick_params(labelsize=12)

    fig.savefig(imd.axis_output_dir + f'/eqwidth_mass.pdf')


plot_mass_vs_eqha()