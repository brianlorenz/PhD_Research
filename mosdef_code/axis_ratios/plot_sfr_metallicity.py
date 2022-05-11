# Plots the balmer decrement vs a vairiety of properies
from operator import methodcaller
from tkinter import font
import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt
from astropy.io import ascii
import numpy as np
from matplotlib.patches import Ellipse
from ellipses_for_plotting import get_ellipse_shapes
import matplotlib as mpl
from plot_vals import *
from dust_model import *
from sympy.solvers import solve
from sympy import Symbol





def plot_sfr_metals(save_name):
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv').to_pandas()
    fig, ax = plt.subplots(figsize = (8,8))

    # Plot the points
    for i in range(len(summary_df)):
        row = summary_df.iloc[i]

        ax.set_ylim(8.2, 8.9)
        ax_y_len = 0.7
        ax.set_xlim(0, 2.6)
        ax_x_len = 2.6
       
        fontsize=14

        cmap = mpl.cm.inferno
        norm = mpl.colors.Normalize(vmin=3, vmax=7) 
        rgba = cmap(norm(row['balmer_dec']))
        
        ellipse_width, ellipse_height = get_ellipse_shapes(ax_x_len, ax_y_len, row['shape'])

      
        ax.errorbar(row['log_use_sfr_median'], row['metallicity_median'], yerr=np.array([[row['err_metallicity_median_low'], row['err_metallicity_median_high']]]).T, color=rgba, marker='None', ls='None')
        ax.add_artist(Ellipse((row['log_use_sfr_median'], row['metallicity_median']), ellipse_width, ellipse_height, facecolor=rgba))
        ax.set_xlabel('SFR', fontsize=fontsize)
        ax.set_ylabel('Metallicity', fontsize=fontsize)
    
    # Plot lines on constant dust
    metal_vals = np.arange(8.1, 8.9, 0.1)
    res = 0.5 * np.ones(len(metal_vals))
    x = Symbol('x')
    # low mass
    A_lambda = 0.7
    re = 0.25
    sfrs = [float(solve(const2 * 10**(a*metal_vals[i]) * (x/(re**2))**(1/n) - A_lambda, x)[0]) for i in range(len(metal_vals))] #Dust
    sfrs=np.array(sfrs)
    log_sfrs = np.log10(sfrs)
    ax.plot(log_sfrs, metal_vals, ls='--', color='black', marker='None', label='$R_\mathrm{eff} = 0.25$, $A_\mathrm{balmer} = 0.7$')
    # high mass
    A_lambda = 2.0
    re = 0.4
    sfrs = [float(solve(const2 * 10**(a*metal_vals[i]) * (x/(re**2))**(1/n) - A_lambda, x)[0]) for i in range(len(metal_vals))] #Dust
    sfrs=np.array(sfrs)
    log_sfrs = np.log10(sfrs)
    ax.plot(log_sfrs, metal_vals, ls='--', color='blue', marker='None', label='$R_\mathrm{eff} = 0.4$, $A_\mathrm{balmer} = 2.0$')
    # high mass
    A_lambda = 2.0
    re = 0.5
    sfrs = [float(solve(const2 * 10**(a*metal_vals[i]) * (x/(re**2))**(1/n) - A_lambda, x)[0]) for i in range(len(metal_vals))] #Dust
    sfrs=np.array(sfrs)
    log_sfrs = np.log10(sfrs)
    ax.plot(log_sfrs, metal_vals, ls='--', color='orange', marker='None', label='$R_\mathrm{eff} = 0.5$, $A_\mathrm{balmer} = 2.0$')

    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Balmer Decrement', fontsize=fontsize)
    ax.tick_params(labelsize=12)
    ax.set_aspect(ellipse_width/ellipse_height)
    ax.legend()



    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/metallicity_sfr.pdf')


plot_sfr_metals('both_sfms_4bin_median_2axis_boot100')