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

iso_df = ascii.read('my_iso.csv').to_pandas()

axisfont = 14
ticksize = 12
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16

fig, ax = plt.subplots(figsize=(8, 7))

# Credit to the Quick Start Make Cluster for some of this code (https://github.com/astropy/PyPopStar/blob/master/docs/Quick_Start_Make_Cluster.ipynb)
idx = np.where(abs(iso_df['mass'] - 1.0) == min(abs(iso_df['mass'] - 1.0)))[0]
turn_on = 0.8
idx_turn_on = np.where(
    abs(iso_df['mass'] - turn_on) == min(abs(iso_df['mass'] - turn_on)))[0]
end_ms = 120


ax.plot(iso_df['m_hst_f127m'][:idx_turn_on[0]] - iso_df['m_hst_f153m'][:idx_turn_on[0]],
        iso_df['m_hst_f153m'][:idx_turn_on[0]], color='red', label='Pre-MS')
ax.plot(iso_df['m_hst_f127m'][idx_turn_on[0]:end_ms] - iso_df['m_hst_f153m'][idx_turn_on[0]:end_ms],
        iso_df['m_hst_f153m'][idx_turn_on[0]:end_ms], color='mediumseagreen', label='Main Seq (MS)')
ax.plot(iso_df['m_hst_f127m'][end_ms:] - iso_df['m_hst_f153m'][end_ms:],
        iso_df['m_hst_f153m'][end_ms:], color='black', label='Post-MS')
ax.plot(iso_df['m_hst_f127m'][idx] - iso_df['m_hst_f153m'][idx],
        iso_df['m_hst_f153m'][idx], '*', ms=15, label='1 $M_\odot$', color='orange')
ax.plot(iso_df['m_hst_f127m'][idx_turn_on] - iso_df['m_hst_f153m'][idx_turn_on],
        iso_df['m_hst_f153m'][idx_turn_on], 'o', ms=10, label=f'MS Turn-on ({turn_on} $M_\odot$)', color='blue')
ax.set_xlabel('F127M - F153M', fontsize=axisfont)
ax.set_ylabel('F153M', fontsize=axisfont)
ax.legend(fontsize=axisfont)
#ax.set_xlim(10000, 2000)
#ax.set_ylim(0.1, 1000)
ax.tick_params(labelsize=ticksize, size=ticks)
ax.invert_yaxis()
fig.savefig('/Users/galaxies-air/Courses/Stars/ps1_mesa/ps1_iso_fig.pdf')
plt.close('all')
