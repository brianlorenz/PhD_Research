# Plot a uvj diagram color coded by axis ratio
import numpy as np
import pandas as pd
from astropy.io import ascii
import matplotlib.pyplot as plt
import matplotlib as mpl
import initialize_mosdef_dirs as imd
import matplotlib.patheffects as path_effects


def plot_uvj_ar():
    '''Plot the uvj diagram colored by axis ratio
    
    '''
    # Read in the axis ratio data
    ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()

    # Read in the uvj colors
    uvj_df = ascii.read(imd.loc_galaxy_uvjs).to_pandas()

    ar_uvj_df = ar_df.merge(uvj_df, left_on=['field', 'v4id'], right_on=['field', 'v4id'])


    # Figure 1 - UVJ WIth everything
    fig, ax = plt.subplots(figsize=(9,8))

    cmap = mpl.cm.viridis 
    norm = mpl.colors.Normalize(vmin=0, vmax=1) 

    for i in range(len(ar_uvj_df)):
        row = ar_uvj_df.iloc[i]
        rgba = cmap(norm(row['use_ratio']))
        ax.plot(row['V_J'], row['U_V'], marker = 'o', ls='None', color=rgba)

    # UVJ diagram lines
    ax.plot((-100, 0.69), (1.3, 1.3), color='black')
    ax.plot((1.5, 1.5), (2.01, 100), color='black')
    xline = np.arange(0.69, 1.5, 0.001)
    yline = xline * 0.88 + 0.69
    ax.plot(xline, yline, color='black')

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Axis Ratio', fontsize=14)
    ax.set_xlabel('V-J', fontsize=14) 
    ax.set_ylabel('U-V', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_xlim(-0.5, 2)
    ax.set_ylim(0, 2.5)
    text2 = ax.text(2.5, 0, 'Edge-on', fontsize=14, color=cmap(norm(0)))
    text1 = ax.text(2.5, 2.5, 'Face-on', fontsize=14, color=cmap(norm(1)))
    text1.set_path_effects([path_effects.Stroke(linewidth=1.7, foreground='black'),
                       path_effects.Normal()])
    text2.set_path_effects([path_effects.Stroke(linewidth=1.7, foreground='black'),
                       path_effects.Normal()])
    fig.savefig(imd.axis_output_dir + f'/uvj_ar.pdf')

    # Figure 2 - three-panel UVJ in mass bins, similar to https://arxiv.org/pdf/2109.14721.pdf

    fig, axarr = plt.subplots(1, 3, figsize=(35,8))


    cmap = mpl.cm.viridis 
    norm = mpl.colors.Normalize(vmin=0, vmax=1) 

    
    mass_95_100_filt = np.logical_and(ar_uvj_df['log_mass']>=9.5, ar_uvj_df['log_mass']<10)
    mass_100_105_filt = np.logical_and(ar_uvj_df['log_mass']>=10, ar_uvj_df['log_mass']<10.5)
    mass_105_110_filt = np.logical_and(ar_uvj_df['log_mass']>=10.5, ar_uvj_df['log_mass']<11)

    mass_filts = [mass_95_100_filt, mass_100_105_filt, mass_105_110_filt]
    titles = ['9.5 <= log(Stellar Mass) < 10', '10 <= log(Stellar Mass) < 10.5', '10.5 <= log(Stellar Mass) < 11']

    count = 0
    for ax in axarr:
        mass_filt = mass_filts[count]
        for i in range(len(ar_uvj_df[mass_filt])):
            

            row = ar_uvj_df[mass_filt].iloc[i]
            rgba = cmap(norm(row['use_ratio']))
            ax.plot(row['V_J'], row['U_V'], marker = 'o', ls='None', color=rgba)

        # UVJ diagram lines
        ax.plot((-100, 0.69), (1.3, 1.3), color='black')
        ax.plot((1.5, 1.5), (2.01, 100), color='black')
        xline = np.arange(0.69, 1.5, 0.001)
        yline = xline * 0.88 + 0.69
        ax.plot(xline, yline, color='black')
        
        ax.set_xlabel('V-J', fontsize=14) 
        ax.set_ylabel('U-V', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_xlim(-0.5, 2)
        ax.set_ylim(0, 2.5)
        ax.set_title(titles[count])
        count += 1
    
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axarr)
    cbar.set_label('Axis Ratio', fontsize=14)
    text2 = ax.text(2.5, 0, 'Edge-on', fontsize=14, color=cmap(norm(0)))
    text1 = ax.text(2.5, 2.5, 'Face-on', fontsize=14, color=cmap(norm(1)))
    text1.set_path_effects([path_effects.Stroke(linewidth=1.7, foreground='black'),
                    path_effects.Normal()])
    text2.set_path_effects([path_effects.Stroke(linewidth=1.7, foreground='black'),
                    path_effects.Normal()])
    fig.savefig(imd.axis_output_dir + f'/uvj_ar_by_mass.pdf')

    # Figure 3 - three-panel UVJ in mass bins, similar to https://arxiv.org/pdf/2109.14721.pdf

    fig, axarr = plt.subplots(1, 3, figsize=(35,8))


    cmap = mpl.cm.viridis 
    norm = mpl.colors.Normalize(vmin=2, vmax=7) 

    
    mass_95_100_filt = np.logical_and(ar_uvj_df['log_mass']>=9.5, ar_uvj_df['log_mass']<10)
    mass_100_105_filt = np.logical_and(ar_uvj_df['log_mass']>=10, ar_uvj_df['log_mass']<10.5)
    mass_105_110_filt = np.logical_and(ar_uvj_df['log_mass']>=10.5, ar_uvj_df['log_mass']<11)

    mass_filts = [mass_95_100_filt, mass_100_105_filt, mass_105_110_filt]
    titles = ['9.5 <= log(Stellar Mass) < 10', '10 <= log(Stellar Mass) < 10.5', '10.5 <= log(Stellar Mass) < 11']

    count = 0
    gal_counter = 0
    for ax in axarr:
        mass_filt = mass_filts[count]
        for i in range(len(ar_uvj_df[mass_filt])):
            

            row = ar_uvj_df[mass_filt].iloc[i]
            if row['ha_flux'] > 0:
                if row['hb_flux'] > 0:
                    row['balmer_dec'] = (row['ha_flux'] / row['hb_flux'])
                    rgba = cmap(norm(row['balmer_dec']))
                    ax.plot(row['V_J'], row['U_V'], marker = 'o', ls='None', color=rgba)
                    gal_counter += 1
                else:
                    continue
            else:
                continue

        # UVJ diagram lines
        ax.plot((-100, 0.69), (1.3, 1.3), color='black')
        ax.plot((1.5, 1.5), (2.01, 100), color='black')
        xline = np.arange(0.69, 1.5, 0.001)
        yline = xline * 0.88 + 0.69
        ax.plot(xline, yline, color='black')
        
        ax.set_xlabel('V-J', fontsize=14) 
        ax.set_ylabel('U-V', fontsize=14)
        ax.tick_params(labelsize=12)
        ax.set_xlim(-0.5, 2)
        ax.set_ylim(0, 2.5)
        ax.set_title(titles[count])
        count += 1
    
    print(gal_counter)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axarr)
    cbar.set_label('Balmer Dec', fontsize=14)
    fig.savefig(imd.axis_output_dir + f'/uvj_balmer_by_mass.pdf')

plot_uvj_ar()