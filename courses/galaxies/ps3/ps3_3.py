import sys
import os
import string
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
import pandas as pd
from scipy.interpolate import interp1d
from astropy.modeling.blackbody import blackbody_nu
from astropy import units as u
import scipy.integrate as integrate
from scipy.optimize import curve_fit

# Location to save figure
figout = '/Users/galaxies-air/Desktop/Galaxies/ps3/'

# Fontsizes for plotting
axisfont = 14
ticksize = 12
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16


def plot_func(part, xdat, ydat, xlab, ylab, xlim=None, ylim=None, line='None', mark='o', hist=False):
    """ Makes a basic plot using inputs.

    Keyword arguments:
    xdat -- the data to plot on the x-axis
    ydat -- the data to plot on the y-axis
    xlab -- the label for the x-axis
    ylab -- the label for the y-axis
    """

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 7))

    # Make the plot
    plt.plot(xdat, ydat[0], color='black', marker=mark, ls=line, label='z = 0')
    plt.plot(xdat, ydat[1], color='orange',
             marker=mark, ls=line, label='z = 2')
    plt.plot(xdat, ydat[2], color='cornflowerblue',
             marker=mark, ls=line, label='z = 4')

    # Set the axis labels
    ax.set_xlabel(xlab, fontsize=axisfont)
    ax.set_ylabel(ylab, fontsize=axisfont)

    # Set the limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Set the tick size
    ax.tick_params(labelsize=ticksize, size=ticks)
    ax.legend(fontsize=axisfont-4)

    # Save the figure
    fig.savefig(figout+'3_'+part)
    plt.close('all')


# Read in the halo data
# z0halo = ascii.read('z0.halos').to_pandas()
# z2halo = ascii.read('z2.halos').to_pandas()
# z4halo = ascii.read('z4.halos').to_pandas()

# halo_tables = [z0halo, z2halo, z4halo]
halo_df = ascii.read(figout+'masses.df').to_pandas()
halo_df.drop('col1', axis=1, inplace=True)
print('Data read')

# Set a binsize
bins = 10**np.arange(8.875, 13.875, 0.25)
# Set the bins to put on the x-axis
plotbins = np.arange(9, 13.75, 0.25)


def findMF(halo, bins):
    # halo_masses = halo['Mvir(4)']
    # Computes the mass function for a given halo
    counts = [np.logical_and(halo > bins[i], halo < bins[i+1])
              for i in range(len(bins)-1)]
    return [np.log10(len(halo[i])/15) for i in counts]


# Compute the IMFs
# IMFs = [findMF(halo, bins) for halo in halo_tables]
IMFs = [findMF(halo_df[col], bins) for col in halo_df.columns]

# Plot for part a
plot_func('a', plotbins, IMFs, 'log(Halo Mass)  (M$_\odot$)',
          'log($\phi$) (Mpc$^{-3}$)', xlim=(8.95, 13.55), line='-', mark='.')

''' PART B '''
tom_mass = np.arange(9, 11.75, 0.25)
tom_data = [-2.2, -2.31, -2.41, -2.54, -2.67, -
            2.76, -2.87, -3.03, -3.13, -3.56, -4.27]
tom_errs_up = [0.05, 0.05, 0.05, 0.06, 0.06, 0.06, 0.07, 0.08, 0.08, 0.1, 0.12]


def plot_func_b(part, xdat, ydat, xlab, ylab, xlim=None, ylim=None, line='None', mark='o', hist=False, linefit=None):
    """ Makes a basic plot using inputs.

    Keyword arguments:
    xdat -- the data to plot on the x-axis
    ydat -- the data to plot on the y-axis
    xlab -- the label for the x-axis
    ylab -- the label for the y-axis
    """

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 7))

    # Make the plo
    plt.plot(xdat, ydat, color='orange',
             marker=mark, ls=line, label='Halos')
    plt.errorbar(tom_mass, tom_data, yerr=tom_errs_up, color='cornflowerblue',
                 marker=mark, ls=line, label='Galaxies, Tomczak+ 2014')
    try:
        plt.plot(plotbins_extended, linefit,
                 color='black', marker=mark, ls=line, label='Fit to low-mass end')
    except Exception as error:
        print(error)

    # Set the axis labels
    ax.set_xlabel(xlab, fontsize=axisfont)
    ax.set_ylabel(ylab, fontsize=axisfont)

    # Set the limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Set the tick size
    ax.tick_params(labelsize=ticksize, size=ticks)
    ax.legend(fontsize=axisfont-4)

    # Save the figure
    fig.savefig(figout+'3_'+part)
    plt.close('all')


plot_func_b('b', plotbins[:], IMFs[1], 'log(Mass)  (M$_\odot$)',
            'log($\phi$) (Mpc$^{-3}$)', xlim=(8.95, 13.55), line='-', mark='.')


def line(xdata, slope, yint):
    return slope*xdata+yint


(slope, yint) = curve_fit(line, np.array(plotbins[2:8]),
                          np.array(IMFs[1][2:8]))[0]

plotbins_extended = np.arange(9, 17, 0.25)

linefit = line(plotbins_extended, slope, yint)

plot_func_b('c', plotbins, IMFs[1], 'log(Mass)  (M$_\odot$)',
            'log($\phi$) (Mpc$^{-3}$)', xlim=(8.95, 17.55), line='-', mark='.', linefit=linefit)


halo_matched_masses = (tom_data-yint)/slope


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
    # ax.legend(fontsize=axisfont-4)

    fig.savefig(figout+name+'.pdf')
    plt.close('all')


fig, ax = plt.subplots(figsize=(8, 7))
plt.plot(tom_mass, np.log10((10**tom_mass)/(10**halo_matched_masses)),
         color='black', marker='o', ls='-')
plot_save(fig, ax, ('log(Stellar Mass)  (M$_\odot$)',
                    'log(Stellar to Halo Mass Ratio)'), '3_c2')
