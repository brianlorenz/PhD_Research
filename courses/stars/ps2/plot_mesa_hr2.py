# plot_mesa_hr.py
# plots the output of mesa onto an hr diagram

import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
import matplotlib.pyplot as plt
from read_mesa import read_history

file_loc = '/Users/galaxies-air/Courses/Stars/ps1_mesa/LOGS/'
new_file_loc = '/Users/galaxies-air/Courses/Stars/ps2_mesa_4/LOGS/'
highmass_data_loc = '/Users/galaxies-air/Courses/Stars/ps2_mesa_highmass/LOGS/'

data_df = read_history(file_loc)
new_data_df = read_history(new_file_loc)
highmass_data_df = read_history(highmass_data_loc)

axisfont = 14
ticksize = 12
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16

fig, ax = plt.subplots(figsize=(8, 7))

ax.plot(10**data_df['log_Teff'], 10**data_df['log_L'],
        color='black', marker=',', ls='-', linewidth=3, label='No Mass Loss')
ax.plot(10**new_data_df['log_Teff'], 10**new_data_df['log_L'],
        color='orange', marker=',', ls='-', linewidth=3, label='Realistic Mass Loss')
ax.plot(10**highmass_data_df['log_Teff'], 10**highmass_data_df['log_L'],
        color='blue', marker=',', ls='-', linewidth=3, label='100x Realistic Mass Loss')
#plt.colorbar(plot, label='log(Age)')
plt.yscale('log')
plt.xscale('log')
ax.set_xlim(10000, 2000)
ax.set_ylim(0.1, 1000)
ax.invert_xaxis()
ax.set_xlabel('T$_{eff}$ (K)', fontsize=axisfont)
ax.set_ylabel('L (L$_\odot$)', fontsize=axisfont)
ax.tick_params(labelsize=ticksize, size=ticks)
ax.legend(fontsize=axisfont-2)
fig.savefig('/Users/galaxies-air/Courses/Stars/ps2/ps2_p4.pdf')
plt.close('all')
