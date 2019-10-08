import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
import sys, os, string
import pandas as pd
from scipy.interpolate import interp1d

figout = '/Users/galaxies-air/Desktop/Galaxies/ps1/'

#Fontsizes for plotting
axisfont = 18
ticksize = 14
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16


part='d'


fitsfiles = [figout + 'stellar_masses.fit', figout + 'total_SFRs.fits', figout + 'metallicities.fits', figout + 'total_sSFRs.fits']

datas = [pd.DataFrame(fits.open(i)[1].data).MEDIAN for i in fitsfiles]
heads = [fits.open(i)[1].header for i in fitsfiles]

#all 3 need good values
idx = np.logical_and(np.logical_and(datas[0]>0, datas[1]>0), datas[2]>-98)
#metallicity+stellar mass good
if (part == 'a'):
    idx = np.logical_and(datas[0]>0, datas[2]>-98)
if (part == 'b'):
    idx = np.logical_and(datas[1]>0, datas[2]>-98)
if (part == 'c'):
    idx = np.logical_and(datas[3]>-98, datas[2]>-98)
if (part == 'd'):
    idx = np.logical_and(datas[0]>0, datas[1]>0)



fig,ax = plt.subplots(figsize=(8,7))

if (part == 'a'):
    ax.plot(datas[0][idx],datas[2][idx],marker='o',ms=0.5,color='black',ls='none') 
    ylab = '12 + log(O/H)'
    xlab = 'log(Stellar Mass) (M$_\\odot$)'
    ax.set_xlim(5.5,13)

if (part == 'b'):
    ax.plot(datas[1][idx],datas[2][idx],marker='o',ms=0.5,color='black',ls='none') 
    ylab = '12 + log(O/H)'
    xlab = 'SFR (M$_\\odot$/year)'
    ax.set_xlim(-0.05,2.5)

if (part == 'c'):
    ax.plot(datas[3][idx],datas[2][idx],marker='o',ms=0.5,color='black',ls='none') 
    ylab = '12 + log(O/H)'
    xlab = 'sSFR (year$^{-1}$)'
    ax.set_xlim(-12.5,-7.5)

if (part == 'd'):
    ax.plot(datas[0][idx],datas[1][idx],marker='o',ms=0.5,color='black',ls='none') 
    ylab = 'SFR (M$_\\odot$/year)'
    xlab = 'log(Stellar Mass) (M$_\\odot$)'
    ax.set_xlim(5.5,13)
    ax.set_ylim(-0.05,2.5)


ax.set_xlabel(xlab,fontsize = axisfont)
ax.set_ylabel(ylab,fontsize = axisfont)
ax.tick_params(labelsize = ticksize, size=ticks)
#ax.legend(fontsize=axisfont-4)
fig.savefig(figout+'2_'+part)
plt.close('all')


