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
from pylab import *


# Location to save figure
figout = '/Users/galaxies-air/Desktop/Galaxies/ps2/'

# Fontsizes for plotting
axisfont = 14
ticksize = 12
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16


# Salpeter function
def salp(m):
    n = m**(-2.35)
    return n

# Plotting function


def plot_func(part, xdat, ydat, xlab, ylab, xlim=None, ylim=None, addkey='', color_range=None, cbarlabel=None, logy=None, logx=None):
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
    try:
        cmap2 = cm.get_cmap('viridis')
        norm = mpl.colors.Normalize(vmin=np.min(
            color_range), vmax=np.max(color_range))
        cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
        cmap.set_array([])

        # [plt.plot(xdat, ydat[i], marker='.', ms=4, ls='None', color=cmap(color_range[i]/np.max(color_range))) for i in np.arange(len(ydat))]
        [plt.plot(xdat, ydat[i], marker='.', ms=4, ls='None',
                  color=cmap2(color_range[i]/np.max(color_range))) for i in np.arange(len(ydat))]
        cbar = fig.colorbar(cmap)
        cbar.set_label(cbarlabel, rotation=270, fontsize=axisfont)
        print(part)
    except Exception as err:
        print(err)
        # Make the plot
        plt.plot(xdat, ydat, color='black', marker='o', ls='None')

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
    # ax.legend(fontsize=axisfont-4)

    # Save the figure
    fig.savefig(figout+'1_'+part+addkey)
    plt.close('all')


'''PART A'''

# Masses - 16 masses equally spaced in log from 0.1 to 100
log_mass = np.arange(-1, 2.2, 0.2)
masses = 10**log_mass

# Find the center of each of the mass bins
mass_center = [(masses[i]+masses[i+1])/2 for i in range(0, 15)]
log_mass_center = [(log_mass[i]+log_mass[i+1])/2 for i in range(0, 15)]
mass_center = np.array(mass_center)
log_mass_center = np.array(log_mass_center)

# Evaluate the salpeter function for every element in binc
salpeter_bin_center = [(masses[i]**(-1.35)-masses[i+1] **
                        (-1.35))/1.35 for i in range(0, 15)]

# Find the sum of salpbinc so we know what to normalize by
norm = np.sum(salpeter_bin_center)

# Probablility weights
weights = salpeter_bin_center/norm

# Generate a stellar population by multiplying the weights by the number of stars
nstars_bins = np.round(weights*10**6)

# Plot for part a
# Set the ydata to log10 of 0.2*nstar, since 0.2 is binsize
plot_func('a', log_mass_center, np.log10(0.2*nstars_bins), 'Log ($M$)',
          'Log ($N$($\Delta$ log $M$))', xlim=(-0.95, 2.05), ylim=(0, 6))


''' PART B '''
# Input a mass, output the Luminosity


def findL(M):
    if M < 0.43:
        L = 0.23*M**2.3
    elif M < 2:
        L = M**4
    elif M < 20:
        L = 1.5*M**3.5
    else:
        L = 3200*M
    return L


# Compute the average luminosity in each bin
luminosity_center = np.array([findL(i) for i in mass_center])

# Comput the mass to light ratio:
mass_to_light = mass_center/luminosity_center

# Plots for part b
plot_func('b', log_mass_center, mass_to_light, 'Log ($M$)',
          'M/L (M/L)$_\odot$', xlim=(-0.95, 2.05))
plot_func('b', log_mass_center, (mass_to_light), 'Log ($M$)',
          'log(M/L) (M/L)$_\odot$', xlim=(-0.95, 2.05), addkey='_log')


'''PART C'''

# Read in the temperature data
temp_df = ascii.read('TL.dat').to_pandas()

# Interpolate the missings values in the L/T relation
temp_interp = interp1d(temp_df['L(L_sun)'], temp_df['T(K)'])

# use the interp function to compute the temperature in each bin
temp_center = temp_interp(luminosity_center)


# Compute a range of plausible luminosities
luminosities = np.e**np.arange(-7, 10.5, 0.1)

# Plots for part c
plot_func('c', luminosities, temp_interp(luminosities), 'L(L$_\odot$)',
          'T(K)', addkey='_lum')

plot_func('c', mass_center, temp_center, 'log(M) (M$_\odot$)',
          'T(K)')

''' PART D '''

# Compute the average lifetime of a star
lifetime_center = 10*mass_center*(luminosity_center**(-1))

# Plots for part d
plot_func('d', log_mass_center, np.log10(lifetime_center), 'log(M) (M$_\odot$)',
          'log(Average Lifetime) (Gyr)')

'''Part E'''

# Mass to light ratio function


def compute_M_L(t):
    """Computes the mass to light ratio, given an age"""
    # Add up the total mass of all stars
    tot_mass = np.sum(nstars_bins*mass_center)

    # Find the total luminosity in each bin
    tot_lumin_bin = luminosity_center*nstars_bins

    # Sum the bins for which the stars are still alive
    tot_lumin = np.sum(tot_lumin_bin[lifetime_center > t])

    # Return the mass to light ratio
    return tot_mass/tot_lumin


# Generate a list of possible times in the universe in Gyr
times = 10**np.arange(-3, 3, 0.01)

# Compute the mass-to-light ratio at each time
mass_to_light_time = [compute_M_L(t) for t in times]

# Plots for part e
plot_func('e', np.log10(times), np.log10(mass_to_light_time), 'log(t) (Gyr)',
          'log(Integrated M/L) (M/L)$_\odot$')

'''Part f'''

# Create the wavelengths to plot over
wavelengths = 10**np.arange(2, 6, 0.001)
nus = (3*10**8)/(wavelengths*10**(-10))

# Create a blackbody spectrum for each bin
blackbody_specs = np.array(
    [blackbody_nu(nus, temp) for temp in temp_center])

# Plot for part f
plot_func('f', np.log10(wavelengths), blackbody_specs, 'log(Wavelength) ($\AA$)',
          'Flux', color_range=temp_center, cbarlabel='Temperature (K)')

'''PART G'''
scaled_specs = np.array([nstars_bins[i]*blackbody_specs[i]
                         for i in range(len(nstars_bins))])


def compute_int_spec(t):
    """Computes the integrated spectrum, given an age"""
    # Sum the bins for which the stars are still alive
    int_spectrum = np.sum(scaled_specs[lifetime_center > t], axis=0)
    int_spectrum_B = int_spectrum[np.logical_and(
        wavelengths < 4450, wavelengths > 4440)]
    int_spectrum_V = int_spectrum[np.logical_and(
        wavelengths < 5510, wavelengths > 5500)]
    print(int_spectrum_B, int_spectrum_V)
    B_V = -2.5*np.log10(int_spectrum_B/int_spectrum_V)

    # Return the integrated spectrum
    return int_spectrum, B_V


# Only compute times where stars die off
integrated_light = [compute_int_spec(t) for t in lifetime_center]

# Grab the spectra and B-V separately
integrated_specs = [integrated_light[i][0]
                    for i in range(len(integrated_light))]
B_Vs = [integrated_light[i][1] for i in range(len(integrated_light))]

# Plot for part g
plot_func('g', np.log10(wavelengths), integrated_specs, 'log(Wavelength) ($\AA$)',
          'Flux', color_range=np.log10(lifetime_center), cbarlabel='log(Age) (Gyr)')

''' PART H '''
# Plot for part h
plot_func('h', np.log10(lifetime_center), B_Vs, 'log(t) (Gyr)',
          'B-V')

# Compute the mass-to-light ratio at each time
mass_to_light_tchange = [compute_M_L(t) for t in lifetime_center]

# Plots for part h2
plot_func('h2', np.log10(mass_to_light_tchange), B_Vs,
          'log(Integrated M/L) (M/L)$_\odot$', 'B-V')


print('Done')
