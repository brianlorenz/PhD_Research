import initialize_mosdef_dirs as imd 
import matplotlib as mpl
from matplotlib.patches import Ellipse
from ellipses_for_plotting import get_ellipse_shapes
import matplotlib.pyplot as plt
import numpy as np
from axis_ratio_helpers import bootstrap_median
from stack_spectra import norm_axis_stack
from matplotlib import patches
import matplotlib.gridspec as gridspec
from astropy.io import ascii
import pandas as pd
from sfms_bins import sfms_slope, sfms_yint
from plot_vals import *

shapes = {'low': '+', 'mid': 'd', 'high': 'o'}
colors = {'sorted0': 'red', 'sorted1': 'blue', 'sorted2': 'orange', 'sorted3': 'mediumseagreen', 'sorted4': 'lightskyblue', 'sorted5': 'darkviolet'}

def plot_sample_split(n_groups, save_name, ratio_bins, starting_points, mass_width, split_width, nbins, sfms_bins, ax='None', fig='fig', plot_sfr_and_ssfr=False):
    """Plots the way that the sample has been divided in mass/ssfr space
    
    variable(str): Which variable to use for the y-axis
    plot_sfr_and_ssfr (boolean): If true, will take the sfms cut and plot it on the ssfr axis
    """

    # If there isn't an axis set, make one - otherwise, use the one provided
    if ax == 'None':
        fig, ax = plt.subplots(figsize=(8,8))
        made_new_axis = True
    else:
        made_new_axis = False
    
    
    group_num = []
    shapes_list = []
    color_list = []
    axis_medians = []
    mass_medians = []
    split_medians = []
    av_medians = []
    err_av_medians = []
    beta_medians = []
    err_beta_medians = []
    mips_flux_medians = []
    err_mips_flux_medians = []
    halpha_snrs = []
    hbeta_snrs = []
    balmer_decs = []
    err_balmer_decs_low = []
    err_balmer_decs_high = []
    balmer_avs = []
    err_balmer_av_lows = []
    err_balmer_av_highs = []
    log_use_ssfr_medians = []
    log_use_sfr_medians = []
    re_medians = []
    err_re_medians = []
    keys = []

    # Figure 1 - Showing how the sample gets cut
    # cmap = mpl.cm.Reds 
    cmap = mpl.cm.gist_heat_r 
    norm = mpl.colors.Normalize(vmin=1, vmax=7) 

    for axis_group in range(n_groups):
        axis_ratio_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_group_dfs/{axis_group}_df.csv').to_pandas()
        emission_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_fits/{axis_group}_emission_fits.csv').to_pandas()
        variable = axis_ratio_df.iloc[0]['split_for_stack']

        axis_ratio_df['balmer_dec'] = (axis_ratio_df['ha_flux'] / axis_ratio_df['hb_flux'])        

        # Adding the balmer decrement measured to summary df:
        balmer_dec = emission_df['balmer_dec'].iloc[0]
        balmer_err_low = emission_df['err_balmer_dec_low'].iloc[0]
        balmer_err_high = emission_df['err_balmer_dec_high'].iloc[0]
        balmer_decs.append(balmer_dec)
        err_balmer_decs_low.append(balmer_err_low)
        err_balmer_decs_high.append(balmer_err_high)
        halpha_snrs.append(emission_df[emission_df['line_name']=='Halpha']['signal_noise_ratio'].iloc[0])
        hbeta_snrs.append(emission_df[emission_df['line_name']=='Hbeta']['signal_noise_ratio'].iloc[0])
        # See Price 2014 for the conversion factors:
        balmer_av = 4.05*1.97*np.log10(balmer_dec/2.86)
        err_balmer_av_low = 1.97*np.sqrt((0.434*((balmer_err_low/balmer_dec)/2.86))**2 + (0.8/4.05)**2)
        err_balmer_av_high = 1.97*np.sqrt((0.434*((balmer_err_high/balmer_dec)/2.86))**2 + (0.8/4.05)**2)
        balmer_avs.append(balmer_av)
        err_balmer_av_lows.append(err_balmer_av_low)
        err_balmer_av_highs.append(err_balmer_av_high)

        # Determine what color and shape to plot with
        axis_median = np.median(axis_ratio_df['use_ratio'])
        mass_median = np.median(axis_ratio_df['log_mass'])
        split_median = np.median(axis_ratio_df[variable])
        av_median = np.median(axis_ratio_df['AV'])
        beta_median = np.median(axis_ratio_df['beta'])
        re_median = np.median(axis_ratio_df['half_light'])


        # We want log_use_sfr and log_use_ssfr in all cases:
        if variable != 'log_use_ssfr':
            log_use_ssfr_median = np.median(axis_ratio_df['log_use_ssfr'])
            log_use_ssfr_medians.append(log_use_ssfr_median)
        if variable != 'log_use_sfr':
            log_use_sfr_median = np.median(axis_ratio_df['log_use_sfr'])
            log_use_sfr_medians.append(log_use_sfr_median)

        #Compute the mips flux, normalized int he same way we normalized the spectra
        # Old way was just axis_ratio_df['mips_flux']
        axis_ratio_df['norm_factor'] = norm_axis_stack(axis_ratio_df['ha_flux'], axis_ratio_df['Z_MOSFIRE'])
        axis_ratio_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_group_dfs/{axis_group}_df.csv', index=False)
        # print(f'Group: {axis_group}, median_norm: {np.median(axis_ratio_df["norm_factor"])}')
        good_mips_idxs = axis_ratio_df['mips_flux'] > -98
        mips_flux_median = np.median(axis_ratio_df[good_mips_idxs]['mips_flux']*axis_ratio_df[good_mips_idxs]['norm_factor'])
        # Better to bootstrap or use MAD?
        err_mips_flux_median = bootstrap_median(axis_ratio_df[good_mips_idxs]['err_mips_flux']*axis_ratio_df[good_mips_idxs]['norm_factor'])



        # Bootstrap errors on the medians
        err_av_median = bootstrap_median(axis_ratio_df['AV'])
        err_beta_median = bootstrap_median(axis_ratio_df['beta'])
        err_re_median = bootstrap_median(axis_ratio_df['half_light'])

        

        # Figure out what axis ratio bin (+1 is since the bins are dividers)
        if len(ratio_bins)+1 == 3:
            if axis_median < ratio_bins[0]:
                shape = shapes['low']
            elif axis_median > ratio_bins[1]:
                shape = shapes['high']
            else:
                shape = shapes['mid']
        
        if len(ratio_bins)+1 == 2:
            if axis_median < ratio_bins[0]:
                shape = shapes['low']
            else:
                shape = shapes['high']
        
        if len(ratio_bins)+1 == 1:
            shape = shapes['high']

        # print(f'Mass median: {mass_median}, SSFR median: {split_median}')
        # Figure out which mass/ssfr bin
        color = 'purple'
        sorted_points = sorted(starting_points)
        if mass_median > sorted_points[0][0] and mass_median < sorted_points[0][0] + mass_width:
            if split_median > sorted_points[0][1] and split_median < sorted_points[0][1] + split_width:
                key = "sorted0"
                # Create a Rectangle patch
                rect = patches.Rectangle((sorted_points[0][0],  sorted_points[0][1]), mass_width, split_width, linestyle='--', linewidth=1, edgecolor=colors[key], facecolor='none')
        if mass_median > sorted_points[1][0] and mass_median < sorted_points[1][0] + mass_width:
            if split_median > sorted_points[1][1] and split_median < sorted_points[1][1] + split_width:
                key = "sorted1"
                rect = patches.Rectangle((sorted_points[1][0],  sorted_points[1][1]), mass_width, split_width, linestyle='--', linewidth=1, edgecolor=colors[key], facecolor='none')
        if nbins > 6 or nbins==4 or len(sorted_points)==6:
            if mass_median > sorted_points[2][0] and mass_median < sorted_points[2][0] + mass_width:
                if split_median > sorted_points[2][1] and split_median < sorted_points[2][1] + split_width:
                    key = "sorted2"
                    rect = patches.Rectangle((sorted_points[2][0],  sorted_points[2][1]), mass_width, split_width, linestyle='--', linewidth=1, edgecolor=colors[key], facecolor='none')
            if mass_median > sorted_points[3][0] and mass_median < sorted_points[3][0] + mass_width:
                if split_median > sorted_points[3][1] and split_median < sorted_points[3][1] + split_width:
                    key = "sorted3"
                    rect = patches.Rectangle((sorted_points[3][0],  sorted_points[3][1]), mass_width, split_width, linestyle='--', linewidth=1, edgecolor=colors[key], facecolor='none')
        if nbins > 12 or len(sorted_points)==6:
            if mass_median > sorted_points[4][0] and mass_median < sorted_points[4][0] + mass_width:
                if split_median > sorted_points[4][1] and split_median < sorted_points[4][1] + split_width:
                    key = "sorted4"
                    rect = patches.Rectangle((sorted_points[4][0],  sorted_points[4][1]), mass_width, split_width, linestyle='--', linewidth=1, edgecolor=colors[key], facecolor='none')
            if mass_median > sorted_points[5][0] and mass_median < sorted_points[5][0] + mass_width:
                if split_median > sorted_points[5][1] and split_median < sorted_points[5][1] + split_width:
                    key = "sorted5"
                    rect = patches.Rectangle((sorted_points[5][0],  sorted_points[5][1]), mass_width, split_width, linestyle='--', linewidth=1, edgecolor=colors[key], facecolor='none')
        if sfms_bins==False:
            color = colors[key]
        else:
            key = f'sorted{axis_group%4}'
            color = 'black'
            rect = patches.Rectangle([0, 0], 0.5, 0.5, linestyle='--', linewidth=1, edgecolor='blue', facecolor='none')

        
        group_num.append(axis_group)
        shapes_list.append(shape)
        color_list.append(color)
        axis_medians.append(axis_median)
        mass_medians.append(mass_median)
        split_medians.append(split_median)
        av_medians.append(av_median)
        err_av_medians.append(err_av_median)
        beta_medians.append(beta_median)
        err_beta_medians.append(err_beta_median)
        mips_flux_medians.append(mips_flux_median)
        err_mips_flux_medians.append(err_mips_flux_median)
        re_medians.append(re_median)
        err_re_medians.append(err_re_median)
        keys.append(key)



        # Set the axis limits
        xlims = (9.0, 11.0)
        if variable == 'log_ssfr' or variable == 'log_halpha_ssfr' or variable == 'log_use_ssfr':
            ylims = (-9.7, -8)
        elif variable == 'eq_width_ha':
            ylims = (0, 600)
        else:
            ylims = (-0.1, 2.6)
            x = np.linspace(8.8, 11.2, 100)
            print(sfms_slope, sfms_yint)
            y1 = sfms_slope*x + sfms_yint
            ax.plot(x, y1, color='grey', ls='--')
            ax.axvline(10, color='grey', ls='--')

            a = -24.0415
            b = 4.1693
            c = -0.1638
            y_sfr = a + b*x + c*x**2
            ax.plot(x, y_sfr, color='black', ls='-.')
        
        # Plots the sfms division line
        add_str=''
        if plot_sfr_and_ssfr == True:
            x = np.linspace(8.8, 11.2, 100)
            print(sfms_slope, sfms_yint)
            y1 = sfms_slope*x + sfms_yint
            y_ssfr = np.log10((10**y1)/(10**x))
            ax.plot(x, y_ssfr, color='black', ls='--')
            # ax.axvline(10, color='black', ls='--')
            add_str = '_bothcuts'
        
        
        ax.set_ylim(ylims)
        ax.set_xlim(xlims)

        # Get the ellipse shapes for plotting
        ellipse_width, ellipse_height = get_ellipse_shapes(xlims[1]-xlims[0], np.abs(ylims[1]-ylims[0]), shape, scale_factor=0.025)

        # plot a black point at the median
        # ax.add_artist(Ellipse((mass_median, split_median), ellipse_width, ellipse_height, facecolor='black', zorder=3))

        


        for i in range(len(axis_ratio_df)):
            row = axis_ratio_df.iloc[i]
            rgba = cmap(norm(row['balmer_dec']))
            # ax.plot(row['log_mass'], row['log_ssfr'], color = rgba, ls='None', marker=shape)
            if row['hb_detflag_sfr'] == 1.0:
                ax.add_artist(Ellipse((row['log_mass'], row[variable]), ellipse_width, ellipse_height, edgecolor=rgba, zorder=2, fill=False, linewidth=2))
            else:
                ax.add_artist(Ellipse((row['log_mass'], row[variable]), ellipse_width, ellipse_height, facecolor=rgba, zorder=2))

       
        # Add the patch to the Axes
        ax.add_patch(rect)



    summary_df = pd.DataFrame(zip(group_num, axis_medians, mass_medians, split_medians, av_medians, err_av_medians, beta_medians, err_beta_medians, mips_flux_medians, err_mips_flux_medians, re_medians, err_re_medians, halpha_snrs, hbeta_snrs, balmer_decs, err_balmer_decs_low, err_balmer_decs_high, balmer_avs, err_balmer_av_lows, err_balmer_av_highs, shapes_list, color_list, keys), columns=['axis_group','use_ratio_median', 'log_mass_median', variable+'_median', 'av_median', 'err_av_median', 'beta_median', 'err_beta_median', 'mips_flux_median', 'err_mips_flux_median', 're_median', 'err_re_median', 'halpha_snr', 'hbeta_snr', 'balmer_dec', 'err_balmer_dec_low', 'err_balmer_dec_high', 'balmer_av', 'err_balmer_av_low', 'err_balmer_av_high', 'shape', 'color', 'key'])    
    if variable != 'log_use_ssfr':
        summary_df['log_use_ssfr_median'] = log_use_ssfr_medians
    if variable != 'log_use_sfr':
        summary_df['log_use_sfr_median'] = log_use_sfr_medians
    if made_new_axis==True:
        summary_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv', index=False)


    ax.set_xlabel(stellar_mass_label, fontsize=16) 
    ax.set_ylabel(variable, fontsize=16)
    if variable=='log_use_sfr':
        ax.set_ylabel(sfr_label, fontsize=16)
    ax.tick_params(labelsize=12)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Balmer Decrement', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    ax.set_aspect(ellipse_width/ellipse_height)
    if made_new_axis==True:
        fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/sample_cut{add_str}.pdf')



