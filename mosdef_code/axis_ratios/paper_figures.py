from axis_ratio_funcs import filter_ar_df
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.patheffects as path_effects

def main():
    sample_f125_vs_f160()

def sample_f125_vs_f160():
    '''Makes an axis ratio plot comparing two methods of determining ar

    Parameters: 
    '''

    ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()


    cmap = mpl.cm.viridis 
    norm = mpl.colors.Normalize(vmin=1.3, vmax=2.8) 

    ar_df, std_ar_diff = filter_ar_df(ar_df, return_std_ar=True)
    


    fig, ax = plt.subplots(figsize=(9,8))


    for i in range(len(ar_df)):
        row = ar_df.iloc[i]
        rgba = cmap(norm(row['z']))

        ax.errorbar(row['F160_axis_ratio'], row['F125_axis_ratio'], xerr=row['F160_err_axis_ratio'], yerr=row['F125_err_axis_ratio'], marker='o', ls='None', color=rgba)
        

    ax.plot((-1,2), (-1, 2), color='black', ls='--', label='F125=F160')
    ax.plot((-1+2*std_ar_diff, 2+2*std_ar_diff), (-1, 2), color='red', ls='-', label='2$\sigma$ deviation')
    ax.plot((-1-2*std_ar_diff, 2-2*std_ar_diff), (-1, 2), color='red', ls='-')

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Redshift', fontsize=14)

    ax.set_xlabel(f'F160 Axis Ratio', fontsize=14) 
    ax.set_ylabel(f'F125 Axis Ratio', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.text(-0.07, -10, 'Edge-on', fontsize=14, zorder=100)
    ax.text(0.95, -10, 'Face-on', fontsize=14, zorder=100)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    ax.legend(fontsize=14)
    fig.savefig(imd.paper_fig_dir + f'/F125_F160_axis_ratio.pdf')


    # Make a second one, this time split by redshift
    fig, [ax_lowz, ax_highz] = plt.subplots(1, 2, figsize=(17,8))


    for i in range(len(ar_df)):
        row = ar_df.iloc[i]
        rgba = cmap(norm(row['z']))
        if row['z']< 2.0:
            ax_lowz.errorbar(row['F160_axis_ratio'], row['F125_axis_ratio'], xerr=row['F160_err_axis_ratio'], yerr=row['F125_err_axis_ratio'], marker='o', ls='None', color=rgba)
        else:
            ax_highz.errorbar(row['F160_axis_ratio'], row['F125_axis_ratio'], xerr=row['F160_err_axis_ratio'], yerr=row['F125_err_axis_ratio'], marker='o', ls='None', color=rgba)

    for ax in [ax_lowz, ax_highz]:
        ax.plot((-1,2), (-1, 2), color='black', ls='--', label='F125=F160')
        ax.plot((-1+2*std_ar_diff, 2+2*std_ar_diff), (-1, 2), color='red', ls='-', label='2$\sigma$ deviation')
        ax.plot((-1-2*std_ar_diff, 2-2*std_ar_diff), (-1, 2), color='red', ls='-')

        ax.set_xlabel(f'F160 Axis Ratio', fontsize=14) 
        ax.set_ylabel(f'F125 Axis Ratio', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.text(-0.07, -10, 'Edge-on', fontsize=14, zorder=100)
        ax.text(0.95, -10, 'Face-on', fontsize=14, zorder=100)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=14)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax_highz)
    cbar.set_label('Redshift', fontsize=14)

    

    
    fig.savefig(imd.paper_fig_dir + f'/F125_F160_axis_ratio_splitz.pdf')


main()