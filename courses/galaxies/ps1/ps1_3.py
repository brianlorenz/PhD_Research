import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
import sys, os, string
import pandas as pd
from scipy.interpolate import interp1d


fitsfile = '/Users/galaxies-air/Desktop/Galaxies/spSpec-51788-0401-161.fit' 

data = fits.open(fitsfile)[0].data
head = fits.open(fitsfile)[0].header
figout = '/Users/galaxies-air/Desktop/Galaxies/ps1/'
filtdatapath = '/Users/galaxies-air/Desktop/Galaxies/ps1/filters/'

flux = data[0]
d1 = data[1]
error = data[2]
d3 = data[3]
d4 = data[4]
#d5 = data[5]
#d6 = data[6]
#d7 = data[7]
#d8 = data[8]

gfilt = ascii.read(filtdatapath + 'g.dat.txt').to_pandas()
ifilt = ascii.read(filtdatapath + 'i.dat.txt').to_pandas()
rfilt = ascii.read(filtdatapath + 'r.dat.txt').to_pandas()
ufilt = ascii.read(filtdatapath + 'u.dat.txt').to_pandas()
zfilt = ascii.read(filtdatapath + 'z.dat.txt').to_pandas()
filts = [gfilt, ifilt, rfilt, ufilt, zfilt]


crval1 = head["crval1"]
crpix1 = head["crpix1"]
#cdelt1 = head["cdelt1"]
naxis1 = head["naxis1"]
coeff0 = head["coeff0"]
coeff1 = head["coeff1"]
dcflag = head["dc-flag"]
exptime = head['exptime']
wavelength = (1.0+np.arange(naxis1)-crpix1)*1.22+crval1+3800# + crval1
wavelength = 10**(coeff0+np.arange(naxis1)*coeff1)

filtmags = []
filtcenters = []
filterrs = []

for filt in filts:
    wavefilt = filt.col1
    y = filt.col2
    f = interp1d(wavefilt, y)
    idx = np.logical_and(wavelength>min(wavefilt),wavelength<max(wavefilt))
    waverange = wavelength[idx]
    fluxrange = flux[idx]
    interpfilt = f(waverange)
    errrange = error[idx]
    totflux = np.sum(fluxrange*waverange*interpfilt)/np.sum(waverange*interpfilt)
    toterr = np.sum(errrange*waverange*interpfilt)/np.sum(waverange*interpfilt)
    appmag = -2.5*np.log10(totflux*10**-17)-21.1
   #appmag = -2.5*np.log10(totflux*10**-17*10**-7)-48.57
    errappmagl = np.abs(-2.5*np.log10((totflux-toterr)*10**-17)-21.1 - appmag)
    errappmagu = np.abs(-2.5*np.log10((totflux+toterr)*10**-17)-21.1 - appmag)
    print(totflux)
    filtmags.append(appmag)
    filterrs.append((errappmagl,errappmagu))
    filtcenters.append(np.median(waverange))



#Fontsizes for plotting
axisfont = 18
ticksize = 14
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16

part = 'c'

fig,ax = plt.subplots(figsize=(8,7))
#ax.plot(wavelength,-2.5*np.log10(data[0]*10**-17)-21.1,zorder=10)
if (part != 'b'):
    ax.plot(wavelength,-2.5*np.log10(data[0]*10**-17)-21.1,zorder=10,color='orange',label='Observed')
if (part == 'd'):
    ax.plot(wavelength/(1+0.1),(-2.5*np.log10((1+0.1)*data[0]*10**-17)-21.1),zorder=10,color='cornflowerblue',label='Rest-Frame')
    ax.plot(gfilt.col1,-gfilt.col2+19,color='green',label='g-band response')
if (part == 'b' or part =='c'):
    ax.errorbar(filtcenters,filtmags,yerr=np.transpose(np.array(filterrs)),ls='None',marker='o',ms=6,color='black',zorder=100,mfc='none',label='Synthesized Photometry')
if (part == 'c'): ax.errorbar(filtcenters,[16.86,15.81,16.23,18.06,15.69],yerr=[0,0,0,0.02,0.01],ls='None',marker='o',ms=6,color='blue',zorder=110,mfc='none',label='SDSS Photometry')
if (part != 'b'): ax.legend(fontsize=legendfont)
ax.set_xlabel('Wavelength ($\AA$)',fontsize = axisfont)
ax.set_ylabel('Apparent Magnitude',fontsize = axisfont)
ax.set_ylim(19,15)
if (part == 'd'): ax.set_ylim(19.05,16)
ax.tick_params(labelsize = ticksize, size=ticks)
fig.savefig(figout+'3_'+part)
plt.close('all')


if (part == 'e'):
    wavefilt = gfilt.col1
    y = gfilt.col2
    f = interp1d(wavefilt, y)
    corwave = wavelength/(1.1)
    idx = np.logical_and(corwave>min(wavefilt),corwave<max(wavefilt))
    waverange = corwave[idx]
    fluxrange = (1+0.1)*flux[idx]
    interpfilt = f(waverange)
    totflux = np.sum(fluxrange*waverange*interpfilt)/np.sum(waverange*interpfilt)
    appmag = -2.5*np.log10(totflux*10**-17)-21.1
   #appmag = -2.5*np.log10(totflux*10**-17*10**-7)-48.57
    print(totflux)

