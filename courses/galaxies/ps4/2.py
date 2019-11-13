# Plot a .fits file

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
import sys
import os
import string
import pandas as pd
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from photutils.isophote import EllipseGeometry
from photutils.isophote import build_ellipse_model
from photutils.isophote import EllipseGeometry
from photutils import EllipticalAperture
from photutils.isophote import Ellipse


# Fontsizes for plotting
axisfont = 24
ticksize = 18
ticks = 8
titlefont = 24
legendfont = 16
textfont = 16


fitsfile = '/Users/galaxies-air/Galfit/kormendy.fits'

figout = '/Users/galaxies-air/Desktop/Galaxies/ps4/'

data = fits.open(fitsfile)[0].data
head = fits.open(fitsfile)[0].header


fig, ax = plt.subplots(figsize=(8, 7))
# ax0,ax1,ax2,ax3,ax4,ax5,ax6,ax7,ax8 = axarr[0,0],axarr[0,1],axarr[0,2],axarr[1,0],axarr[1,1],axarr[1,2],axarr[2,0],axarr[2,1],axarr[2,2]
ax.set_xlabel('x (Pixels)', fontsize=axisfont)
ax.set_ylabel('y (Pixels)', fontsize=axisfont)
cb = ax.imshow(data, vmin=10**11, vmax=2*10**14,
               cmap='gray_r', norm=LogNorm())
ax.tick_params(labelsize=ticksize, size=ticks)
colorbar = plt.colorbar(cb)
colorbar.ax.set_ylabel('Flux (erg/s/cm^2)', rotation=270, fontsize=axisfont-8)
colorbar.ax.tick_params(labelsize=ticksize-4)


geometry = EllipseGeometry(x0=350, y0=350, sma=20, eps=0, pa=0.)
aper = EllipticalAperture((geometry.x0, geometry.y0),
                          geometry.sma, geometry.sma*(1 - geometry.eps), geometry.pa)
ellipse = Ellipse(data, geometry)
isolist = ellipse.fit_image()
#model_image = build_ellipse_model(data.shape, isolist)
#residual = data - model_image

smas = np.linspace(10, 200, 5)
for sma in smas:
    iso = isolist.get_closest(sma)
    x, y, = iso.sampled_coordinates()
    ax.plot(x, y, color='red')

plt.tight_layout()
fig.savefig(figout+'kormendy_iso.pdf')
plt.close('all')

# Making the radial cute
len_cut = 150
radial_cut = data[350][350:(350+len_cut)]
pixels_axis = np.arange(len_cut)

# Fitting a line


def line(xdata, slope, yint):
    return slope*xdata+yint


cutval = 60
(slope, yint) = curve_fit(
    line, pixels_axis[cutval:], np.log10(radial_cut[cutval:]))[0]

linefit = line(pixels_axis, slope, yint)

print(f'Central Surface Brightness: {linefit[0]}')

# Figure setup
fig, ax = plt.subplots(figsize=(8, 7))

mark = '.'
xlab = 'Radius (pixels)'
ylab = 'Flux (erg/s/cm^2)'
# xlim = (8.95, 10.05)
# ylim = (-12, -8)

plt.plot(pixels_axis, radial_cut, color='black',
         marker=mark, label='Data')
plt.plot(pixels_axis, 10**linefit, color='red',
         marker=None, alpha=0.9, label='Fit to Exponential')
plt.scatter(0, 10**linefit[0], color='red',
            marker='*', label='Central Flux', s=40)


# Set the axis labels
ax.set_xlabel(xlab, fontsize=axisfont)
ax.set_ylabel(ylab, fontsize=axisfont)

ax.set_yscale('log')
# ax.set_xscale('log')

# Set the limits
# ax.set_xlim(xlim)
# ax.set_ylim(ylim)

ax.legend(fontsize=axisfont-4)

# Set the tick size
ax.tick_params(labelsize=ticksize, size=ticks)

plt.tight_layout()

# Save the figure
fig.savefig(figout+'profile_kor_2.pdf')
plt.close('all')
