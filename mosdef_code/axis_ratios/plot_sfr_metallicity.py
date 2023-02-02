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

        ax.set_ylim(8.2, 8.95)
        ax_y_len = 0.75
        if plot_ssfr==True:
            ax.set_xlim(-9.6, -8.1)
            ax_x_len = 1.5
        else:
            ax.set_xlim(0, 2.6)
            ax_x_len = 2.6
       
        fontsize=full_page_axisfont

        cmap = mpl.cm.inferno
        # norm = mpl.colors.Normalize(vmin=3, vmax=5.5) 
        # rgba = cmap(norm(row['balmer_dec']))
        norm = mpl.colors.Normalize(vmin=0.5, vmax=2.5) 
        rgba = cmap(norm(row['balmer_av']))
        
        ellipse_width, ellipse_height = get_ellipse_shapes(ax_x_len, ax_y_len, row['shape'])

        if plot_ssfr == True:
            x_points = row['log_use_ssfr_median']
            ax.set_xlabel('log(sSFR)', fontsize=fontsize)
        else:
            x_points = row['log_use_sfr_median']
            ax.set_xlabel(sfr_label, fontsize=fontsize)
            if plot_re==True:
                x_points = np.log10(10**row['log_use_sfr_median'] / row['re_median'])
                ax.set_xlabel('log(SFR/R_e)', fontsize=fontsize)
        ax.errorbar(x_points, row['metallicity_median'], yerr=np.array([[row['err_metallicity_median_low'], row['err_metallicity_median_high']]]).T, color=rgba, marker='None', ls='None', zorder=3)
        zorder=15-i
        ax.add_artist(Ellipse((x_points, row['metallicity_median']), ellipse_width, ellipse_height, facecolor=rgba, edgecolor='black', zorder=zorder))
        ax.set_ylabel('12 + log(O/H)', fontsize=fontsize)
    
    # Plot lines on constant dust
    metal_vals = np.arange(8.1, 9.0, 0.1)
    x = Symbol('x')
   
    def get_slope(x1, x2, y1, y2):
        slope = (y2-y1) /(x2-x1) 
        print(slope)
        return slope

    # low mass
    A_lambda = 0.85
    re = 0.25
    sfrs = [float(solve(const2 * 10**(a*metal_vals[i]) * (x/(re**2))**(1/n) - A_lambda, x)[0]) for i in range(len(metal_vals))] #Dust
    sfrs=np.array(sfrs)
    log_sfrs = np.log10(sfrs)
    if plot_ssfr == True:
        log_mass = 9.75
        x_plot = np.log10(sfrs/(10**log_mass))
        label = '$R_\mathrm{eff} = 0.25$, $A_\mathrm{balmer} = 0.85$' + f', mass={log_mass}'
    else:
        x_plot = log_sfrs
        if plot_re==True:
            x_plot = np.log10(10**log_sfrs/re)
        label = '$R_\mathrm{eff} = 0.25$, $A_\mathrm{balmer} = 0.85$'
    ax.plot(x_plot, metal_vals, ls='--', color='#8E248C', marker='None', zorder=2)
    get_slope(x_plot[0], x_plot[-1], metal_vals[0], metal_vals[-1])

    # high mass
    A_lambda = 1.9
    re = 0.41
    sfrs = [float(solve(const2 * 10**(a*metal_vals[i]) * (x/(re**2))**(1/n) - A_lambda, x)[0]) for i in range(len(metal_vals))] #Dust
    sfrs=np.array(sfrs)
    log_sfrs = np.log10(sfrs)
    if plot_ssfr == True:
        log_mass = 10.25
        x_plot = np.log10(sfrs/(10**log_mass))
        label = '$R_\mathrm{eff} = 0.4$, $A_\mathrm{balmer} = 1.9$' + f', mass={log_mass}'
    else:
        x_plot = log_sfrs
        label = '$R_\mathrm{eff} = 0.4$, $A_\mathrm{balmer} = 1.9$'
        if plot_re==True:
            x_plot = np.log10(10**log_sfrs/re)
    ax.plot(x_plot, metal_vals, ls='--', color='#FF640A', marker='None', zorder=2)
    get_slope(x_plot[0], x_plot[-1], metal_vals[0], metal_vals[-1])

    # high mass
    if plot_ssfr == True:
        pass
    else:
        pass
        # A_lambda = 2.0
        # re = 0.5
        # sfrs = [float(solve(const2 * 10**(a*metal_vals[i]) * (x/(re**2))**(1/n) - A_lambda, x)[0]) for i in range(len(metal_vals))] #Dust
        # sfrs=np.array(sfrs)
        # log_sfrs = np.log10(sfrs)
        # if plot_re==True:
        #     log_sfrs = np.log10(10**np.log10(sfrs)/re)
        # ax.plot(log_sfrs, metal_vals, ls='--', color='skyblue', marker='None', label='$R_\mathrm{eff} = 0.5$, $A_\mathrm{balmer} = 2.0$')


    # mass/metal/sfr relation
    def compute_metals(log_mass, fm_s, re):
        '''
        Parameters:
        log_mass: Log stellar mass
        fm_s: log sfrs (array)
        '''
        fm_m = (log_mass-10)*np.ones(len(fm_s))
        fm_metals = fundamental_plane(fm_m, fm_s)
        add_str2 = ''
        if plot_sanders==True:
            fm_metals = sanders_plane(log_mass, fm_s)
            add_str2 = '_sanders'
        if plot_ssfr == True:
            x_plot = np.log10(10**fm_s/(10**log_mass))
        elif plot_re == True:
            x_plot = np.log10(10**fm_s/(re))
        else:
            x_plot = fm_s
        return x_plot, fm_metals, add_str2

    fm_s = np.arange(0, 3, 0.1)
    log_mass = 9.55
    re = 0.25
    x_plot, fm_metals_lowm_bot, add_str2 = compute_metals(log_mass, fm_s, re) 
    # ax.plot(x_plot, fm_metals_lowm_bot, ls='--', color='black', marker='None', label=f'Stellar Mass = {log_mass}, {add_str2}')
    log_mass = 9.8
    x_plot, fm_metals_lowm_top, add_str2 = compute_metals(log_mass, fm_s, re) 
    # ax.plot(x_plot, fm_metals_lowm_top, ls='--', color='black', marker='None', label=f'Stellar Mass = {log_mass}, {add_str2}')
    ax.fill_between(x_plot, fm_metals_lowm_bot, fm_metals_lowm_top, color='black', alpha=0.35, zorder=1)
    get_slope(x_plot[10], x_plot[16], fm_metals_lowm_bot[10], fm_metals_lowm_bot[16])


    log_mass = 10.2
    re = 0.4
    x_plot, fm_metals_highm_bot, add_str2 = compute_metals(log_mass, fm_s, re) 
    # ax.plot(x_plot, fm_metals_highm_bot, ls='--', color='blue', marker='None', label=f'Stellar Mass = {log_mass}, {add_str2}')
    log_mass = 10.35
    x_plot, fm_metals_highm_top, add_str2 = compute_metals(log_mass, fm_s, re) 
    # ax.plot(x_plot, fm_metals_highm_top, ls='--', color='blue', marker='None', label=f'Stellar Mass = {log_mass}, {add_str2}')
    ax.fill_between(x_plot, fm_metals_highm_bot, fm_metals_highm_top, color='black', alpha=0.2, zorder=1)
    get_slope(x_plot[10], x_plot[16], fm_metals_highm_bot[10], fm_metals_highm_bot[16])

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    # cbar.set_label(balmer_label, fontsize=fontsize)
    cbar.set_label('A$_\mathrm{balmer}$', fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    ax.tick_params(labelsize=full_page_axisfont)
    ax.set_aspect(ellipse_width/ellipse_height)
    # ax.legend()

    ax.text(1.345, 8.21, 'Low M$_*$ FMR', fontsize=18, rotation=315)
    ax.text(1.91, 8.33, 'High M$_*$ FMR', fontsize=18, rotation=315)
    ax.text(0.18, 8.60, 'A$_\mathrm{balmer} = 0.85$', fontsize=16, rotation=308, color='#8E248C')
    ax.text(0.80, 8.71, 'A$_\mathrm{balmer} = 1.9$', fontsize=16, rotation=308, color='#FF640A')
    
    ax.plot([0],[0],ls='--',color='dimgrey',marker='None',label='Dust Model')
    ax.legend(fontsize=16)
    plt.setp(ax.get_yticklabels()[0], visible=False)   


    if plot_ssfr==True:
        add_str = '_ssfr'
    elif plot_re==True:
        add_str = '_divre'
    elif plot_sanders==True:
        add_str = '_sanders'
    else:
        add_str = ''

    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/metallicity_sfr{add_str}.pdf',bbox_inches='tight')


# plot_sfr_metals('whitaker_sfms_boot100')
# plot_sfr_metals('whitaker_sfms_boot100', plot_sanders=True)
# plot_sfr_metals('whitaker_sfms_boot100', plot_re=True)


def plot_sfr_times_metals(save_name):
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv').to_pandas()
    fig, ax = plt.subplots(figsize = (8,8))

    # Plot the points
    for i in range(len(summary_df)):
        row = summary_df.iloc[i]

        ax.set_ylim(8, 9)
        ax_y_len = 1
        
        ax.set_xlim(9.5, 10.5)
        ax_x_len = 1
    
        fontsize=14

        cmap = mpl.cm.inferno
        # norm = mpl.colors.Normalize(vmin=3, vmax=5.5) 
        # rgba = cmap(norm(row['balmer_dec']))
        norm = mpl.colors.Normalize(vmin=0.5, vmax=2.5) 
        rgba = cmap(norm(row['balmer_av']))
        
        ellipse_width, ellipse_height = get_ellipse_shapes(ax_x_len, ax_y_len, row['shape'])

        x_points = row['log_mass_median']
        ax.set_xlabel(stellar_mass_label, fontsize=fontsize)
        row_err = np.array([[row['err_metallicity_median_low'], row['err_metallicity_median_high']]]).T
        ax.errorbar(x_points, row['metallicity_median']*row['log_use_sfr_median'], yerr=row_err, color=rgba, marker='None', ls='None', zorder=3)
        zorder=15-i
        ax.add_artist(Ellipse((x_points, row['metallicity_median']), ellipse_width, ellipse_height, facecolor=rgba, edgecolor='black', zorder=zorder))
        ax.set_ylabel('Metallicity*log SFR', fontsize=fontsize)


        # Plot lines on constant dust
        metal_vals = np.arange(8.1, 9.0, 0.1)
        x = Symbol('x')
    
        # low mass
        A_lambda = 0.85
        re = 0.25
        sfrs = [float(solve(const2 * 10**(a*metal_vals[i]) * (x/(re**2))**(1/n) - A_lambda, x)[0]) for i in range(len(metal_vals))] #Dust
        sfrs=np.array(sfrs)
        log_sfrs = np.log10(sfrs)
        x_plot = log_sfrs*metal_vals
        print(x_plot)
        # ax.plot(x_plot, metal_vals, ls='--', color='#8E248C', marker='None', zorder=2)

    ax.tick_params(labelsize=14)
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/metallicity_times_sfr.pdf',bbox_inches='tight')

# plot_sfr_metals('norm_1_sn5_filtered', plot_sanders=True)

