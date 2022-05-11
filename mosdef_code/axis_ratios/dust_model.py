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

n = 1.4
a = 1.77
b = -17.96
const = 100
const2 = 2.5*10**(-17)

def dust_model(metallicity, sfr, re, mass):
    # A_lambda = const * (10**(a * metallicity + b)) * ((sfr / (2 * np.pi * (re**2)))**(1/n))
    A_lambda = const2 * 10**(a*metallicity) * (sfr/(re**2))**(1/n)
    # A_lambda = const * (a * metallicity + b) * (sfr)**(1/n)
    # A_lambda = const * metallicity * (sfr)**(1/n)
    return A_lambda

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

# test_dust_model()
# plot_dust_model('both_sfms_4bin_median_2axis_boot100')
# plot_dust_model('both_sfms_4bin_median_2axis_boot100_retest')
