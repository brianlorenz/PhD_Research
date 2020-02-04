# Reads mesa's History file into a pandas table

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from read_mesa import read_history

file_loc = '/Users/galaxies-air/Courses/Stars/ps1_mesa/LOGS/'

data_df = read_history(file_loc)

axisfont = 14
ticksize = 12
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16

fig, ax = plt.subplots(figsize=(8, 7))

plot = ax.scatter(10**data_df['log_Teff'], 10**data_df['log_L'],
                  c=np.log10(data_df['star_age']), marker='o')
plt.colorbar(plot, label='log(Age)')
plt.yscale('log')
plt.xscale('log')
ax.set_xlim(10000, 2000)
ax.set_ylim(0.1, 1000)
ax.invert_xaxis()
ax.set_xlabel('T$_{eff}$ (K)', fontsize=axisfont)
ax.set_ylabel('L (L$_\odot$)', fontsize=axisfont)
ax.tick_params(labelsize=ticksize, size=ticks)
fig.savefig('/Users/galaxies-air/Courses/Stars/ps1_mesa/ps1_mesa_fig.pdf')
plt.close('all')
