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
figout = '/Users/galaxies-air/Desktop/Galaxies/ps2/'

# Fontsizes for plotting
axisfont = 14
ticksize = 12
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16


def exp_sfr(t):
    # T in Gyr, a in Msun/year
    return 1.0067857*10**11*np.e**(-t)


def const_sfr(t):
    return (10**(11))/5+0*t


def rising_sfr(t):
    return 10**t


def plot_func(part, xdat, ydat, ydat2, xlab, ylab, xlim=None, ylim=None, addkey='', color_range=None, cbarlabel=None, logy=None, logx=None):
    """ Makes a basic plot using inputs.

    Keyword arguments:
    xdat -- the data to plot on the x-axis
    ydat -- the data to plot on the y-axis
    xlab -- the label for the x-axis
    ylab -- the label for the y-axis
    """

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 7))

    # Assume that there are multiple data in y
    ax.plot(xdat, ydat, color='cornflowerblue', marker=None, lw=3,
            ls='-', label='Expotential SFR')
    ax.plot(xdat, ydat2, color='orange', marker=None, lw=3,
            ls='-', label='Constant SFR')

    # Set the axis labels
    ax.set_xlabel(xlab, fontsize=axisfont)
    ax.set_ylabel(ylab, fontsize=axisfont)

    # Set the limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Check if log scale
    if logy:
        plt.yscale('log')
    if logx:
        plt.xscale('log')

    # Set the tick size
    ax.tick_params(labelsize=ticksize, size=ticks)
    ax.legend(fontsize=axisfont)

    # Save the figure
    fig.savefig(figout+'2_'+part+addkey)
    plt.close('all')


time = np.arange(0, 13.8, 0.01)
exp_sfrs = exp_sfr(time)
const_sfrs = const_sfr(time)
rising_sfrs = rising_sfr(time)


plot_func('a', time, exp_sfrs/10**9, const_sfrs/10**9, 'Time (Gyr)',
          'SFR (M$_\odot$ yr$^{-1}$)', xlim=(-0.05, max(time)+0.05))

''' PART B '''


def get_ssfr(t, sfr_func, sfrs):
    masses = [integrate.quad(sfr_func, 0, i)[0] for i in t]
    return sfrs/masses


exp_ssfrs = get_ssfr(time, exp_sfr, exp_sfrs)
const_ssfrs = get_ssfr(time, const_sfr, const_sfrs)
rising_ssfrs = get_ssfr(time, rising_sfr, rising_sfrs)

plot_func('b', time, exp_ssfrs/10**9, const_ssfrs/10**9, 'Time (Gyr)',
          'sSFR (yr$^{-1}$)', xlim=(-0.05, max(time)+0.05), ylim=(0, 2/10**9))


''' PART C '''
SFR_conv = 1.26*10**41
plot_func('c', time, (exp_sfrs/10**9)*SFR_conv, (const_sfrs/10**9)*SFR_conv, 'Time (Gyr)',
          'H$\\alpha$ Luminosity (erg/s)', xlim=(-0.05, max(time)+0.05))

''' Part E '''
# Salpeter function


def salp(m):
    n = (2*m)**(-2.35)
    return n


def newimf(m):
    if m < 0.5:
        n = (2*m)**(-1.3)
    else:
        n = (2*m)**(-2.35)
    return n


ms = 10**np.arange(-1, 2, 0.01)

salps = np.array([salp(i) for i in ms])
newimfs = np.array([newimf(i) for i in ms])


plot_func('e', np.log10(ms),  np.log10(newimfs), np.log10(salps), 'log M (M$_\odot$)',
          'dn/dM', xlim=(-0.99, 2.01))

print(integrate.quad(salp, 0.08, 100))
print(integrate.quad(newimf, 0.08, 100))
print(integrate.quad(newimf, 0.08, 100)[0]/integrate.quad(salp, 0.08, 100)[0])

print('Done')
