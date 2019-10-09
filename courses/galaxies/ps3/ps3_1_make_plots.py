import sys
import os
import string
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
import pandas as pd

loc = '/Users/galaxies-air/Desktop/Galaxies/ps3/'

spectra_df = pd.read_csv(loc+'ssp_spectra.df')
'''Reads in a dataframe with columns:
wavelength - ranges from 91 to 10^8 Angstrom
'z' + metallicity + '_t' + age - spectrum of ssp with that metallicity and age
metallicity can be 'low', 'mid', or 'high' (-0.5, 0, 0.5 Zsun)
age can be 1Gyr, 2, 5, or 14
'''

ew_df = pd.read_csv(loc+'ew.df')
'''Reads in a dataframe with columns:
'z' + metallicity + '_t' + age + '_' + line + '_ew'
where line is either 'Hb' or "Mgb"
'''


def plot_save(fig, ax, labels, name, xlim=None, ylim=None, logs=(0, 0)):
    '''
    Adds axis labels, sets limits, and saves the plot
    labels - tuple of strings - (xlabel, ylabel)
    name - string - name to save the file under
    xlim - tuple - xlimits of the graph
    ylim - tuple
    logs - tuple - (xlog, ylog). Set value to 1 if you want that scale to be log
    '''

    axisfont = 14
    ticksize = 12
    ticks = 8
    titlefont = 24
    legendfont = 14
    textfont = 16

    ax.set_xlabel(labels[0], fontsize=axisfont)
    ax.set_ylabel(labels[1], fontsize=axisfont)

    if logs[0] == 1:
        plt.xscale('log')
    if logs[1] == 1:
        plt.yscale('log')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.tick_params(labelsize=ticksize, size=ticks)
    ax.legend(fontsize=axisfont-4)

    fig.savefig(loc+name+'.pdf')
    plt.close('all')


wavelength = spectra_df['wavelength']

ages = ('1', '2', '5', '14')
colors = ('black', 'orange', 'blue', 'green')
age_labels = [f'{i} Gyr' for i in ages]

axis_labels = ('Wavelength ($\AA$)', 'Flux')

# Plots a range of ages at solar metallicity
fig, ax = plt.subplots(figsize=(8, 7))
for age, color, label in zip(ages, colors, age_labels):
    plt.plot(wavelength, spectra_df['zmid_t'+age],
             color=color, marker=None, ls='-', label=label)
plot_save(fig, ax, axis_labels, '1_a_ages', logs=(1, 0))

# Keys to retrieve which metallicity to plot
z_keys = ('low', 'mid', 'high')
z_labels = [f'log(Z$_\odot$) = {i}' for i in (-0.5, 0, 0.5)]

# Plots galaxies at 5Gyr with a range of metallicities


def make_z_plot(name, xlim=None, ylim=None, logs=(1, 0)):
    fig, ax = plt.subplots(figsize=(8, 7))
    for z, color, label in zip(z_keys, colors[:-1], z_labels):
        plt.plot(wavelength, spectra_df['z'+z+'_t5'],
                 color=color, marker=None, ls='-', label=label)
    plot_save(fig, ax, axis_labels, name, xlim=xlim, ylim=ylim, logs=logs)


make_z_plot('1_a_zs')

# Zoomed in version of the above plot
make_z_plot('1_a_zs_zoom', xlim=(5000, 5400),
            ylim=(3*10**-17, 8*10**-16), logs=(0, 1))


axis_labels_ew_age = ('Age (Gyr)', 'Equivalent Width of Mgb')

fig, ax = plt.subplots(figsize=(8, 7))
for z, color, label in zip(z_keys, colors, z_labels):

    z_ages = [ew_df['z'+z+'_t'+age+'_Mgb'] for age in ages]

    plt.plot([1, 2, 5, 14], z_ages,
             color=color, marker=None, ls='-', label=label)
plot_save(fig, ax, axis_labels_ew_age, '1_b')


axis_labels_ews = ('Equivalent Width of Mgb', 'Equivalent Width of Hb')

keys = ['z'+z+'_t'+age for z in z_keys for age in ages]

Mgb_measurements = [ew_df[key+'_Mgb'] for key in keys]
Hb_measurements = [ew_df[key+'_Hb'] for key in keys]

fig, ax = plt.subplots(figsize=(8, 7))
plt.plot(Hb_measurements[:4], Mgb_measurements[:4],
         color='black', marker='o', ls='-')
plt.plot(Hb_measurements[4:8], Mgb_measurements[4:8],
         color='black', marker='o', ls='-')
plt.plot(Hb_measurements[8:], Mgb_measurements[8:],
         color='black', marker='o', ls='-')

plt.plot(Hb_measurements[::4], Mgb_measurements[::4],
         color='black', marker='o', ls='-')
plt.plot(Hb_measurements[1::4], Mgb_measurements[1::4],
         color='black', marker='o', ls='-')
plt.plot(Hb_measurements[2::4], Mgb_measurements[2::4],
         color='black', marker='o', ls='-')
plt.plot(Hb_measurements[3::4], Mgb_measurements[3::4],
         color='black', marker='o', ls='-')

plot_save(fig, ax, axis_labels_ews, '1_c')
