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





def plot_sfr_metals(save_name, plot_ssfr=False, plot_re=False, plot_sanders=False):
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv').to_pandas()
    fig, ax = plt.subplots(figsize = (8,8))

    # Plot the points
    for i in range(len(summary_df)):
        row = summary_df.iloc[i]

        ax.set_ylim(8.2, 8.9)
        ax_y_len = 0.7
        if plot_ssfr==True:
            ax.set_xlim(-9.6, -8.1)
            ax_x_len = 1.5
        else:
            ax.set_xlim(0, 2.6)
            ax_x_len = 2.6
       
        fontsize=14

        cmap = mpl.cm.inferno
        norm = mpl.colors.Normalize(vmin=3, vmax=7) 
        rgba = cmap(norm(row['balmer_dec']))
        
        ellipse_width, ellipse_height = get_ellipse_shapes(ax_x_len, ax_y_len, row['shape'])

        if plot_ssfr == True:
            x_points = row['log_use_ssfr_median']
            ax.set_xlabel('log(sSFR)', fontsize=fontsize)
        else:
            x_points = row['log_use_sfr_median']
            ax.set_xlabel('log(SFR)', fontsize=fontsize)
            if plot_re==True:
                x_points = np.log10(10**row['log_use_sfr_median'] / row['re_median'])
                ax.set_xlabel('log(SFR/R_e)', fontsize=fontsize)
        ax.errorbar(x_points, row['metallicity_median'], yerr=np.array([[row['err_metallicity_median_low'], row['err_metallicity_median_high']]]).T, color=rgba, marker='None', ls='None')
        ax.add_artist(Ellipse((x_points, row['metallicity_median']), ellipse_width, ellipse_height, facecolor=rgba))
        ax.set_ylabel('Metallicity', fontsize=fontsize)
    
    # Plot lines on constant dust
    metal_vals = np.arange(8.1, 8.9, 0.1)
    res = 0.5 * np.ones(len(metal_vals))
    x = Symbol('x')
    # low mass
    A_lambda = 0.7
    re = 0.3
    sfrs = [float(solve(const2 * 10**(a*metal_vals[i]) * (x/(re**2))**(1/n) - A_lambda, x)[0]) for i in range(len(metal_vals))] #Dust
    sfrs=np.array(sfrs)
    log_sfrs = np.log10(sfrs)
    if plot_ssfr == True:
        log_mass = 9.75
        x_plot = np.log10(sfrs/(10**log_mass))
        label = '$R_\mathrm{eff} = 0.25$, $A_\mathrm{balmer} = 0.7$' + f', mass={log_mass}'
    else:
        x_plot = log_sfrs
        if plot_re==True:
            x_plot = np.log10(10**log_sfrs/re)
        label = '$R_\mathrm{eff} = 0.25$, $A_\mathrm{balmer} = 0.7$'
    ax.plot(x_plot, metal_vals, ls='--', color='black', marker='None', label=label)
    # high mass
    A_lambda = 2.0
    re = 0.4
    sfrs = [float(solve(const2 * 10**(a*metal_vals[i]) * (x/(re**2))**(1/n) - A_lambda, x)[0]) for i in range(len(metal_vals))] #Dust
    sfrs=np.array(sfrs)
    log_sfrs = np.log10(sfrs)
    if plot_ssfr == True:
        log_mass = 10.25
        x_plot = np.log10(sfrs/(10**log_mass))
        label = '$R_\mathrm{eff} = 0.4$, $A_\mathrm{balmer} = 2.0$' + f', mass={log_mass}'
    else:
        x_plot = log_sfrs
        label = '$R_\mathrm{eff} = 0.4$, $A_\mathrm{balmer} = 2.0$'
        if plot_re==True:
            x_plot = np.log10(10**log_sfrs/re)
    ax.plot(x_plot, metal_vals, ls='--', color='blue', marker='None', label=label)
    
    # high mass
    if plot_ssfr == True:
        pass
    else:
        A_lambda = 2.0
        re = 0.5
        sfrs = [float(solve(const2 * 10**(a*metal_vals[i]) * (x/(re**2))**(1/n) - A_lambda, x)[0]) for i in range(len(metal_vals))] #Dust
        sfrs=np.array(sfrs)
        log_sfrs = np.log10(sfrs)
        if plot_re==True:
            log_sfrs = np.log10(10**np.log10(sfrs)/re)
        ax.plot(log_sfrs, metal_vals, ls='--', color='skyblue', marker='None', label='$R_\mathrm{eff} = 0.5$, $A_\mathrm{balmer} = 2.0$')


    # mass/metal/sfr relation
    fm_s = np.arange(0, 3, 0.1)
    log_mass = 9.45
    fm_m = (log_mass-10)*np.ones(len(fm_s))
    fm_metals = fundamental_plane(fm_m, fm_s)
    add_str2 = ''
    if plot_sanders==True:
        fm_metals = sanders_plane(log_mass, fm_s)
        print(fm_metals)
        add_str2 = '_sanders'
    if plot_ssfr == True:
        x_plot = np.log10(10**fm_s/(10**log_mass))
    else:
        x_plot = fm_s
    ax.plot(x_plot, fm_metals, ls='--', color='red', marker='None', label=f'Stellar Mass = {log_mass}, {add_str2}')
    # mass/metal/sfr relation
    fm_s = np.arange(0, 3, 0.1)
    log_mass = 10.2
    fm_m = (log_mass-10)*np.ones(len(fm_s))
    fm_metals = fundamental_plane(fm_m, fm_s)
    add_str2 = ''
    if plot_sanders==True:
        fm_metals = sanders_plane(log_mass, fm_s)
        add_str2 = '_sanders'
    if plot_ssfr == True:
        x_plot = np.log10(10**fm_s/(10**log_mass))
    else:
        x_plot = fm_s
    ax.plot(x_plot, fm_metals, ls='--', color='orange', marker='None', label=f'Stellar Mass = {log_mass}, {add_str2}')
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Balmer Decrement', fontsize=fontsize)
    ax.tick_params(labelsize=12)
    ax.set_aspect(ellipse_width/ellipse_height)
    ax.legend()


    if plot_ssfr==True:
        add_str = '_ssfr'
    elif plot_re==True:
        add_str = '_divre'
    elif plot_sanders==True:
        add_str = '_sanders'
    else:
        add_str = ''

    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/metallicity_sfr{add_str}.pdf')


# plot_sfr_metals('both_sfms_4bin_median_2axis_boot100')
# plot_sfr_metals('both_sfms_4bin_median_2axis_boot100', plot_re=True)
# plot_sfr_metals('both_sfms_4bin_median_2axis_boot100', plot_sanders=True)
# plot_sfr_metals('both_sfms_4bin_median_2axis_boot100', plot_ssfr=True)