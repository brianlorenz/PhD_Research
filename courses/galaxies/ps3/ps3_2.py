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

    # If hist
    if hist:
        plt.hist(xdat, 50, weights=[1/np.sum(xdat)
                                    for i in xdat], facecolor='blue', alpha=0.5)

    # Make the plot
    else:
        plt.plot(xdat, ydat, color='black', marker=mark, ls=line)

    # Set the axis labels
    ax.set_xlabel(xlab, fontsize=axisfont)
    ax.set_ylabel(ylab, fontsize=axisfont)

    # Set the limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Set the tick size
    ax.tick_params(labelsize=ticksize, size=ticks)
    # ax.legend(fontsize=axisfont-4)

    # Save the figure
    fig.savefig(figout+'2_'+part)
    plt.close('all')

# Computes the evolution of metallicity given constraints in part a


def z_evo(t):
    return -0.5*np.log(1 - 0.054*t)


# Generate a range of times from 0 to 20 Gyr
times = np.arange(0, 20, 0.01)

# Compute teh metallicities
zs = [z_evo(i) for i in times]

# Plot for part a
plot_func('a', times, zs, 'Time (Gyr)',
          'Z (Z$_\odot$)', xlim=(-0.05, 20.05), line='-', mark='.')

# The distribution of stars reflects the distribution of metallicities
stars_dist = np.array(zs)[times <= 13]

# Normalize the distribution:
stars_dist_norm = stars_dist/np.sum(stars_dist)

# Plot for part b
plot_func('b', stars_dist, stars_dist_norm, 'Z (Z$_\odot$)',
          'Relative fraction of Stars', xlim=(-0.05, 0.7), hist=True)


def exp_sfr(t):
    return np.e**(-t/11)


def z_evo_c(t):
    gas_frac = 1-0.092*integrate.quad(exp_sfr, 0, t)[0]
    return -0.5*np.log(gas_frac)


# Compute teh metallicities
zs2 = [z_evo_c(i) for i in times]

# Plot for part c
plot_func('c', times, zs2, 'Time (Gyr)',
          'Z (Z$_\odot$)', xlim=(-0.05, 20.05), line='-', mark='.')

# The distribution of stars reflects the distribution of metallicities
stars_dist_2 = np.array(zs2)[times <= 13]

# Normalize the distribution:
stars_dist_norm_2 = stars_dist_2/np.sum(stars_dist_2)

# Plot for part b
plot_func('c2', stars_dist_2, stars_dist_norm_2, 'Z (Z$_\odot$)',
          'Relative fraction of Stars', xlim=(-0.05, 0.7), hist=True)
