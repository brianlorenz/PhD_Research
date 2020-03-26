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
log_dir_solar = mr.MesaLogDir(
    '/Users/galaxies-air/Courses/Stars/ps4/ps4_mesa_z0/LOGS')
log_dir_lowz = mr.MesaLogDir(
    '/Users/galaxies-air/Courses/Stars/ps4/ps4_mesa_z-2/LOGS')
profile_solar = log_dir_solar.profile_data(817)
profile_lowz = log_dir_lowz.profile_data(918)


axisfont = 14
ticksize = 12
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16


def plot_mesa(x_solar, y_solar, x_lowz, y_lowz, xlabel, ylabel, savename):
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(x_solar, y_solar,
            color='blue', label='Solar Metallicity')
    ax.plot(x_lowz, y_lowz, color='orange', label='Z=0.01Z$_\odot$')

    ax.set_xlabel(xlabel, fontsize=axisfont)
    ax.set_ylabel(ylabel, fontsize=axisfont)
    ax.set_yscale('log')
    ax.tick_params(labelsize=ticksize, size=ticks)
    ax.legend(fontsize=axisfont)
    fig.savefig(f'/Users/galaxies-air/Courses/Stars/ps4/{savename}.pdf')
    plt.close('all')


plot_mesa(profile_solar.R, profile_solar.P, profile_lowz.R,
          profile_lowz.P, 'Radius (R$_\odot$)', 'Pressure (Ba)', 'ps4_mesa_P_R')
plot_mesa(profile_solar.R, profile_solar.T, profile_lowz.R,
          profile_lowz.T, 'Radius (R$_\odot$)', 'Temperature (K)', 'ps4_mesa_T_R')
