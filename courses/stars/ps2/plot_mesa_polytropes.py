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
import mesa_reader as mr

# Read MESA Output
log_dir = mr.MesaLogDir('/Users/galaxies-air/Courses/Stars/ps1_mesa/LOGS')
profile = log_dir.profile_data(800)

# Read polytrope files
poly_colnames = ['xi', 'theta', 'd_theta']

n15_file = '/Users/galaxies-air/code/courses/stars/ps2/output/Poly-n1.5.txt'
n15_df = ascii.read(n15_file).to_pandas()
n15_df.columns = poly_colnames

n3_file = '/Users/galaxies-air/code/courses/stars/ps2/output/Poly-n3.0.txt'
n3_df = ascii.read(n3_file).to_pandas()
n3_df.columns = poly_colnames

axisfont = 14
ticksize = 12
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16

fig, ax = plt.subplots(figsize=(8, 7))

ax.plot(profile.R, profile.P, color='black', label='MESA model')
ax.plot(n15_df['xi']/np.max(n15_df['xi']),
        n15_df['theta']**(1+1/1.5)*1.75*10**17, color='orange', label='n=1.5 Polytrope')
ax.plot(n3_df['xi']/np.max(n3_df['xi']),
        n3_df['theta']**(1+1/3)*1.75*10**17, color='blue', label='n=3 Polytrope')
ax.set_xlabel('Radius (R$_\odot$)', fontsize=axisfont)
ax.set_ylabel('Pressure (Ba)', fontsize=axisfont)
ax.tick_params(labelsize=ticksize, size=ticks)
ax.legend(fontsize=axisfont)
fig.savefig('/Users/galaxies-air/Courses/Stars/ps2_mesa/ps2_mesa_P_R.pdf')
plt.close('all')


dlogP_dlogRho = np.diff(np.log10(profile.P))/np.diff(np.log10(profile.Rho))

fig, ax = plt.subplots(figsize=(8, 7))

ax.plot(profile.R[:-1], dlogP_dlogRho, color='black', label='MESA model')
# plt.yscale('log')
# plt.xscale('log')
ax.set_xlabel('Radius (R$_\odot$)', fontsize=axisfont)
ax.set_ylabel('d log(P) / d log($\\rho$)', fontsize=axisfont)
ax.set_ylim(1, 2)
ax.tick_params(labelsize=ticksize, size=ticks)
ax.legend(fontsize=axisfont)
fig.savefig(
    '/Users/galaxies-air/Courses/Stars/ps2_mesa/ps2_mesa_logPlogRho_R.pdf')
plt.close('all')
