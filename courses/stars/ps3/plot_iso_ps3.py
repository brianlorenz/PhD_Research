# plot_isochrone.py
# Plots the output from pypopstar_ps1.py

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
import matplotlib.pyplot as plt
from pypopstar_ps3 import evo_model_name, logAges

savenames = [f'iso_{logAge}_{evo_model_name[i]}.csv' for logAge in logAges for i in range(2)]

iso_df = ascii.read('my_iso.csv').to_pandas()
iso_df = ascii.read('my_iso.csv').to_pandas()
iso_df = ascii.read('my_iso.csv').to_pandas()
iso_df = ascii.read('my_iso.csv').to_pandas()


axisfont = 14
ticksize = 12
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16


def plot_iso(ax, iso_df, color):
    """

    Parameters:
    ax (plt.axis): axis to plot the isochrone onto
    iso_df (pd.DataFrame): dataframe with isochrone data


    Returns:
    """
    ax.plot(iso_df['m_hst_f127m'] - iso_df['m_hst_f153m'],
            iso_df['m_hst_f153m'], color=color, label='Pre-MS')


fig, ax = plt.subplots(figsize=(8, 7))

plot_iso(ax, )


ax.set_xlabel('F127M - F153M', fontsize=axisfont)
ax.set_ylabel('F153M', fontsize=axisfont)
ax.legend(fontsize=axisfont)
#ax.set_xlim(10000, 2000)
#ax.set_ylim(0.1, 1000)
ax.tick_params(labelsize=ticksize, size=ticks)
ax.invert_yaxis()
fig.savefig('/Users/galaxies-air/Courses/Stars/ps1_mesa/ps1_iso_fig.pdf')
plt.close('all')
