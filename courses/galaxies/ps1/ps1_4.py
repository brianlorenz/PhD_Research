import numpy as np
import matplotlib.pyplot as plt
from astropy.io import ascii, fits
import sys, os, string
import pandas as pd
from scipy.interpolate import interp1d
from scipy.integrate import simps

figout = '/Users/galaxies-air/Desktop/Galaxies/ps1/'

#Fontsizes for plotting
axisfont = 18
ticksize = 14
ticks = 8
titlefont = 24
legendfont = 14
textfont = 16


part='c'

L = np.arange(-2.01,1.01,0.01)
L = 10**L
a = [-1.5,-1.0,-0.5]
norm = [8.95,37.1,0]
norm = [1,1,1]
phiL = [norm*((L[i]**a)*np.exp(-L[i])) for i in np.arange(1,(len(L)-1))]
phiL = np.transpose(phiL)

#M = np.arange(-2.01,1.01,0.01)
#M = -2.5*np.log10(M)
#M = M[np.logical_not(np.isnan(M))]
if (part == 'c'): L = np.log10(L)
phiM = [0.4*np.log(10)*(10**(-0.4*L[i]))**(a[0]+1)*np.exp(-10**(-0.4*L[i])) for i in np.arange(1,(len(L)-1))]
phiM2 = [0.4*np.log(10)*(10**(-0.4*L[i]))**(a[1]+1)*np.exp(-10**(-0.4*L[i])) for i in np.arange(1,(len(L)-1))]
phiM3 = [0.4*np.log(10)*(10**(-0.4*L[i]))**(a[2]+1)*np.exp(-10**(-0.4*L[i])) for i in np.arange(1,(len(L)-1))]
phiM = np.transpose(phiM)
phiM2 = np.transpose(phiM2)
phiM3 = np.transpose(phiM3)

area = np.sum(phiL[0])/len(phiL[0])
print(area)

colors = ['red','black','blue']
labls = ['$\\alpha = -1.5$', '$\\alpha = -1.0$', '$\\alpha = -0.5$']

fig,ax = plt.subplots(figsize=(8,7))

if (part == 'b'):
    [ax.plot(np.log10(L[1:(len(L)-1)]),np.log10(phiL[j]),marker='o',ms=0.01,color=colors[j],label=labls[j]) for j in range(3)]
    xlab = 'Log ($L$/$L_*$)'
    ylab = 'Log $\phi(L)$'

if (part == 'b3'):
    sum0 = [np.sum(phiL[0][i:-1])/i for i in range(0,len(phiL[0]))]
    sum1 = [np.sum(phiL[1][i:-1])/i for i in range(0,len(phiL[1]))]
    sum2 = [np.sum(phiL[2][i:-1])/i for i in range(0,len(phiL[2]))]
    sums = [sum0,sum1,sum2]
    [ax.plot(np.log10(L[1:(len(L)-1)]),np.log10(sums[j]),marker='o',ms=0.01,color=colors[j],label=labls[j]) for j in range(3)]
    ylab = 'Log N(>$L$)'
    xlab = 'Log ($L$/$L_*$)'

if (part == 'c'):
    ax.plot(L[1:(len(L)-1)],np.log10(phiM),marker='o',ms=0.01,color=colors[0],label=labls[0])
    ax.plot(L[1:(len(L)-1)],np.log10(phiM2),marker='o',ms=0.01,color=colors[1],label=labls[1])
    ax.plot(L[1:(len(L)-1)],np.log10(phiM3),marker='o',ms=0.01,color=colors[2],label=labls[2])
    xlab = 'M - M* '
    ylab = 'Log $\phi(M)$'

if (part == 'd'):
    [ax.plot(np.log10(L[1:(len(L)-1)]),np.log10(phiL[j]),marker='o',ms=0.01,color=colors[j],label=labls[j]) for j in range(3)]
    xlab = 'Log ($L$/$L_*$)'
    ylab = 'Log $\phi(L)$'

if (part == 'd2'):
    Lm = L[1:(len(L)-1)]
    [ax.plot(np.log10(Lm),np.log10(Lm*phiL[j]),marker='o',ms=0.01,color=colors[j],label=labls[j]) for j in range(3)]
    #[ax.plot(np.log10(Lm),np.log10(Lm*phiL[j]),marker='o',ms=0.01,color=colors[j],label=labls[j]) for j in range(3)]
    xlab = 'Log ($M_S$/$M_{S*}$)'
    ylab = 'Log ($M_S \phi(M_S)$)'


ax.set_xlabel(xlab,fontsize = axisfont)
ax.set_ylabel(ylab,fontsize = axisfont)
ax.tick_params(labelsize = ticksize, size=ticks)
ax.legend(fontsize=axisfont-4)
fig.savefig(figout+'4_'+part)
plt.close('all')
