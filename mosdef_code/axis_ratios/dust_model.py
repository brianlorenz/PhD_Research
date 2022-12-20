from curses import meta
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import numpy as np
from matplotlib.patches import Ellipse
from ellipses_for_plotting import get_ellipse_shapes
import sys
from scipy.optimize import curve_fit
from plot_vals import *

n = 1.4
a = 1.77
b = -17.96
# a = 2.15
# b = -21.19
# const = 100
const2 = 2.5*10**(-17)
# const2 = 2.5*10**(b)
const3 = 1.086*100*((1 / (2 * np.pi))**(1/n)) * (10**(b))
print(const3)
print(const2)

def dust_model(metallicity, sfr, re, mass):
    # A_lambda = const * (10**(a * metallicity + b)) * ((sfr / (2 * np.pi * (re**2)))**(1/n))
    A_lambda = const2 * 10**(a*metallicity) * (sfr/(re**2))**(1/n)
    # A_lambda = const * (a * metallicity + b) * (sfr)**(1/n)
    # A_lambda = const * metallicity * (sfr)**(1/n)
    return A_lambda

fm_const = 8.90
fm_const_m = 0.37
fm_const_s = -0.14
fm_const_m2 = -0.19
fm_const_ms = 0.12
fm_const_s2 = -0.054
def fundamental_plane(m, s):
    """ From Mannucci, 

    Parameters:
    m (float): log(stellar_mass) - 10
    s (float): log(SFR) 
    (solar units)
    """
    metallicity = fm_const + fm_const_m*m + fm_const_s*s + fm_const_m2*(m**2) + fm_const_ms*m*s + fm_const_s2*(s**2)
    return metallicity

def fundamental_metallictiy(log_mass,log_sfr):
    u = log_mass - 0.32*log_sfr
    x = u-10
    print(u)
    metallicity = 8.9 + 0.47*x
    return metallicity


def sanders_plane(log_mass, log_sfr):
    u60 = log_mass - 0.6*log_sfr
    y = u60 - 10
    metallicity = 8.8 + (0.188*y) + (-0.22 * y**2) + (-0.0531 * y**3)
    return metallicity

def test_dust_model():
    metallicity = 8.7
    sfr = 10**(0.81)
    re = 0.26
    print(dust_model(metallicity, sfr, re, 0))

    sfr = 3.1*sfr
    metallicity = 0.977*metallicity
    print(metallicity)
    print(dust_model(metallicity, sfr, re, 0))


def plot_dust_model(save_name):
    fig, ax = plt.subplots(figsize=(8,8))
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv').to_pandas()
    metallicitys = summary_df['metallicity_median']
    sfrs = 10**summary_df['log_use_sfr_median']
    res = summary_df['re_median']
    mass = summary_df['log_mass_median']
    balmer_avs = summary_df['balmer_av']
    dust_vals = dust_model(metallicitys, sfrs, res, mass)
    print(dust_vals)
    # sys.exit()
    summary_df['dust_model_av'] = dust_vals
    ax_xlim = (0,5)
    ax_ylim = (0,5)
    ax_x_len = ax_xlim[1]-ax_xlim[0]
    ax_y_len = ax_ylim[1]-ax_ylim[0]
    for i in range(len(summary_df)):
        row = summary_df.iloc[i]

        ellipse_width, ellipse_height = get_ellipse_shapes(ax_x_len, ax_y_len, row['shape'])

        cmap = mpl.cm.inferno
        norm = mpl.colors.Normalize(vmin=-9.3, vmax=-8.1) 
        norm = mpl.colors.Normalize(vmin=0, vmax=2.6) 
        rgba = cmap(norm(row['log_use_sfr_median']))
        # ax.plot(row['dust_model_av'], row['balmer_av'], marker='o', color=rgba)
        ax.add_artist(Ellipse((row['dust_model_av'], row['balmer_av']), ellipse_width, ellipse_height, facecolor=rgba))
        ax.plot([-1, 100], [-1, 100], ls='--', color='red')
    ax.set_xlim(ax_xlim)
    ax.set_ylim(ax_ylim)
    ax.set_xlabel('Dust Model')
    ax.set_ylabel('Balmer AV')
    # plt.show()
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/dust_model.pdf')

def plot_on_fmr(save_name):
    fig, ax = plt.subplots(figsize=(8,8))
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv').to_pandas()
    metallicities = summary_df['metallicity_median']
    log_sfrs = summary_df['log_use_sfr_median']
    log_masses = summary_df['log_mass_median']
    high_mass = log_masses>10
    u60s = log_masses - (0.6 * log_sfrs)

    ax.plot(u60s, metallicities, color=light_color, ls='None', marker='o', markersize=8, label='Low mass')
    ax.plot(u60s[high_mass], metallicities[high_mass], color=dark_color, ls='None', marker='o', markersize=8, label='High mass')
    
    ax.set_xlim(8.0, 11.25)
    ax.set_ylim(7.9, 8.9)

    
    def sanders_plane2(u60s):
        y = u60s - 10
        metallicity = 8.8 + (0.188*y) - (0.22 * y**2) - (0.0531 * y**3)
        return metallicity
    u60s_xrange = np.arange(8.2, 11, 0.02)
    yrange = sanders_plane2(u60s_xrange)
    ax.plot(u60s_xrange, yrange, color='black', ls='--', marker='None', label='Sanders FMR')
    ax.tick_params(labelsize=16)
    ax.legend(fontsize=16)
    ax.set_xlabel('$\\mu_{60}$ = log(mass) - 0.6*log(SFR)', fontsize=16)
    ax.set_ylabel(metallicity_label, fontsize=16)
    scale_aspect(ax)
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/fmr.pdf')
    



# plot_on_fmr('whitaker_sfms_boot100')
# test_dust_model()
# plot_dust_model('both_sfms_4bin_median_2axis_boot100')