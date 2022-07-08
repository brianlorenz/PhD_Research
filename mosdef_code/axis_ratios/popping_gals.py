import os
from scipy import stats
from astropy.io import ascii
import numpy as np
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
from axis_ratio_funcs import read_interp_axis_ratio, filter_ar_df, read_filtered_ar_df
from scipy.optimize import curve_fit
import pandas as pd
from plot_vals import *
import matplotlib as mpl


def read_popping():
    # Read the dataframes
    popping_df = ascii.read(imd.mosdef_dir + '/axis_ratio_data/popping_gals.csv').to_pandas()
    ar_df = read_filtered_ar_df()

    # Compute dust-to-gas ratios
    popping_df['dust_to_gas'] = (10**popping_df['log_dust_mass']) / (10**popping_df['log_gas_mass'])
    ar_df['log_use_sfr'] = np.log10(ar_df['use_sfr'])
    ar_df['balmer_dec'] = ar_df['ha_flux']/ar_df['hb_flux']

    # Match the dataframes row by row
    for i in range(len(popping_df)):
        v4id = popping_df.iloc[i]['v4id']
        row = ar_df[ar_df['v4id']==v4id]
        if i==0:
            match_df = row
        match_df = pd.concat([match_df, row], ignore_index=True)
    total_df = match_df.merge(popping_df)
    return total_df

def plot_popping():
    popping_df = read_popping()


    ###SFR Metallicity, color coded by balmer dec
    cmap = mpl.cm.inferno 
    norm = mpl.colors.Normalize(vmin=3, vmax=7) 

    fig, ax = plt.subplots(figsize = (8,8))
    for i in range(len(popping_df)):
        row = popping_df.iloc[i]
        rgba = cmap(norm(row['balmer_dec']))
        ax.errorbar(row['log_use_sfr'], row['metallicity'], yerr=row['err_metallicity'], ls='None', marker='o', color=rgba, mew=1, mec='black', markersize=8)
    ax.set_xlabel(sfr_label, fontsize = 16)
    ax.set_ylabel(metallicity_label, fontsize = 16)
    ax.tick_params(labelsize=16)
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(balmer_label, fontsize=16)
    
    scale_aspect(ax)
    fig.savefig(imd.axis_output_dir + '/popping_sfr_metallicity.pdf')
    plt.close('all')


    ### dust to gas ratio
    cmap = mpl.cm.inferno 
    norm = mpl.colors.Normalize(vmin=-3, vmax=-1.5) 
    popping_df['log_dust_to_gas'] = np.log10(popping_df['dust_to_gas'])
    fig, ax = plt.subplots(figsize = (8,8))
    for i in range(len(popping_df)):
        row = popping_df.iloc[i]
        rgba = cmap(norm(row['log_dust_to_gas']))
        if row['err_log_gas_mass'] > 1:
            # rgba = 'blue'
            pass
        ax.errorbar(row['log_use_sfr'], row['metallicity'], yerr=row['err_metallicity'], ls='None', marker='o', color=rgba, markersize=8, mew=1, mec='black')
    ax.set_xlabel(sfr_label, fontsize = 16)
    ax.set_ylabel(metallicity_label, fontsize = 16)
    ax.tick_params(labelsize=16)
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('log(dust_to_gas_ratio)', fontsize=16)
    
    scale_aspect(ax)
    fig.savefig(imd.axis_output_dir + '/popping_dust_to_gas.pdf')
    plt.close('all')

            
        


plot_popping()