from re import A
from typing import AsyncContextManager
from initialize_mosdef_dirs import check_and_make_dir
from matplotlib.pyplot import axis
from numpy.core.fromnumeric import var
from stack_spectra import *
from fit_emission import fit_emission
from stack_continuum import stack_all_continuum, plot_all_spec_with_cont
import matplotlib as mpl
from matplotlib.patches import Ellipse
from ellipses_for_plotting import get_ellipse_shapes
from plot_overlaid_spectra import plot_overlaid_spectra
from axis_group_metallicities import add_metals_to_summary_df, plot_metals, measure_metals, plot_group_metals_compare, plot_mass_metal
from balmer_avs import plot_balmer_stellar_avs
import random
import matplotlib.gridspec as gridspec
import time

random.seed(3284923)


#18 bins, 2 mass 3 ssfr
# mass_width = 0.8
# split_width = 0.5
# starting_points = [(9.3, -9.1), (9.3, -8.6), (10.1, -9.1), (10.1, -8.6), (9.3, -9.6), (10.1, -9.6)]
# ratio_bins = [0.4, 0.7]
# nbins = 18
# split_by = 'ssfr'

# 6 bins, 2 mass 1 ssfr
# mass_width = 0.8
# split_width = 4
# starting_points = [(9.3, -11), (10.1, -11)]
# ratio_bins = [0.4, 0.7]
# nbins = 6
# split_by = 'ssfr'


shapes = {'low': '+', 'mid': 'd', 'high': 'o'}
colors = {'sorted0': 'red', 'sorted1': 'blue', 'sorted2': 'orange', 'sorted3': 'mediumseagreen', 'sorted4': 'lightskyblue', 'sorted5': 'darkviolet'}


def stack_all_and_plot_all(param_class):
    '''performs all t he steps to get this group plotted
    
    Parameters:
    nbins (int): Number of total bins, calculated as mass_bins*ssfr_bins*axis_ratio_bins
    split_by (str): y-axis variable, typically either ssfr or eq_width_ha
    stack_type (str): mean or median, what to use when making the stacks pixel by pixel
    only_plot (boolean): If set to 1, only do plotting, skip over clustering and stacking
    '''
    # only do all the functions if run_stack is True
    if param_class.run_stack == False:
        return
    nbins = param_class.nbins
    save_name = param_class.save_name
    split_by = param_class.split_by
    stack_type = param_class.stack_type
    only_plot = param_class.only_plot
    mass_width = param_class.mass_width
    split_width = param_class.split_width
    starting_points = param_class.starting_points
    ratio_bins = param_class.ratio_bins
    sfms_bins = param_class.sfms_bins
    print(f'Running stack {save_name}. Making just the plots: {only_plot}')
    time_start = time.time()
    if only_plot==False:
        setup_new_stack_dir(save_name)
        stack_axis_ratio(mass_width, split_width, starting_points, ratio_bins, save_name, split_by, stack_type, sfms_bins)
        stack_all_continuum(nbins, save_name=save_name)
        time_stack = time.time()
        print(f'All stacking took {time_stack-time_start}')
        plot_all_spec_with_cont(nbins, save_name) # This is where the normalized cont is saved
        for axis_group in range(nbins):
            fit_emission(0, 'cluster_norm', constrain_O3=False, axis_group=axis_group, save_name=save_name, scaled='False', run_name='False')
        time_emfit = time.time()
        print(f'Emission fitting took {time_emfit-time_stack}')
    plot_sample_split(nbins, save_name, ratio_bins, starting_points, mass_width, split_width, nbins, sfms_bins)
    plot_overlaid_spectra(save_name, plot_cont_sub=True)
    plot_metals(save_name)
    measure_metals(nbins, save_name)
    plot_group_metals_compare(nbins, save_name)
    plot_mass_metal(nbins, save_name)
    add_metals_to_summary_df(save_name, metal_column='O3N2_metallicity')
    plot_balmer_dec(save_name, nbins, split_by, y_var='balmer_dec', color_var=split_by)
    plot_balmer_dec(save_name, nbins, split_by, y_var='balmer_dec', color_var='metallicity')
    plot_balmer_dec(save_name, nbins, split_by, y_var='av', color_var=split_by)
    plot_balmer_dec(save_name, nbins, split_by, y_var='beta', color_var=split_by)
    plot_balmer_dec(save_name, nbins, split_by, y_var='metallicity', color_var=split_by)
    plot_balmer_dec(save_name, nbins, split_by, y_var='metallicity', color_var='log_use_sfr')
    plot_balmer_dec(save_name, nbins, split_by, y_var='log_use_sfr', color_var='metallicity')
    plot_balmer_dec(save_name, nbins, split_by, y_var='balmer_dec', color_var='log_use_ssfr')
    plot_balmer_dec(save_name, nbins, split_by, y_var='mips_flux', color_var=split_by)
    plot_balmer_stellar_avs(save_name)
    time_end = time.time()
    print(f'Total program took {time_end-time_start}')

def setup_new_stack_dir(save_name):
    """Sets up the directory with all the necessary folders"""
    check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}')
    sub_dirs = ['composite_images', 'composite_seds', 'cont_subs', 'conts', 'emission_fits', 'emission_images', 'spectra', 'spectra_images', 'group_dfs']
    for name in sub_dirs:
        check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_{name}')
    check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts')    
    return



def plot_sample_split(n_groups, save_name, ratio_bins, starting_points, mass_width, split_width, nbins, sfms_bins):
    """Plots the way that the sample has been divided in mass/ssfr space
    
    variable(str): Which variable to use for the y-axis"""

    fig, ax = plt.subplots(figsize=(8,8))

    
    
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
    keys = []

    # Figure 1 - Showing how the sample gets cut
    cmap = mpl.cm.viridis 
    norm = mpl.colors.Normalize(vmin=2, vmax=7) 

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
            shape = shapes['low']

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
        if nbins > 6 or nbins==4:
            if mass_median > sorted_points[2][0] and mass_median < sorted_points[2][0] + mass_width:
                if split_median > sorted_points[2][1] and split_median < sorted_points[2][1] + split_width:
                    key = "sorted2"
                    rect = patches.Rectangle((sorted_points[2][0],  sorted_points[2][1]), mass_width, split_width, linestyle='--', linewidth=1, edgecolor=colors[key], facecolor='none')
            if mass_median > sorted_points[3][0] and mass_median < sorted_points[3][0] + mass_width:
                if split_median > sorted_points[3][1] and split_median < sorted_points[3][1] + split_width:
                    key = "sorted3"
                    rect = patches.Rectangle((sorted_points[3][0],  sorted_points[3][1]), mass_width, split_width, linestyle='--', linewidth=1, edgecolor=colors[key], facecolor='none')
        if nbins > 12:
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
            key = f'sorted{axis_group%6}'
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
        keys.append(key)



        # Set the axis limits
        xlims = (9.0, 11.0)
        if variable == 'log_ssfr' or variable == 'log_halpha_ssfr' or variable == 'log_use_ssfr':
            ylims = (-9.7, -8)
        elif variable == 'eq_width_ha':
            ylims = (0, 600)
        else:
            ylims = (-0.1, 2.6)
        ax.set_ylim(ylims)
        ax.set_xlim(xlims)

        # Get the ellipse shapes for plotting
        ellipse_width, ellipse_height = get_ellipse_shapes(xlims[1]-xlims[0], np.abs(ylims[1]-ylims[0]), shape)

        # plot a black point at the median
        ax.add_artist(Ellipse((mass_median, split_median), ellipse_width, ellipse_height, facecolor='black', zorder=3))

        


        for i in range(len(axis_ratio_df)):
            row = axis_ratio_df.iloc[i]
            rgba = cmap(norm(row['balmer_dec']))
            # ax.plot(row['log_mass'], row['log_ssfr'], color = rgba, ls='None', marker=shape)
        
            ax.add_artist(Ellipse((row['log_mass'], row[variable]), ellipse_width, ellipse_height, facecolor=rgba, zorder=2))

       
        # Add the patch to the Axes
        ax.add_patch(rect)



    summary_df = pd.DataFrame(zip(group_num, axis_medians, mass_medians, split_medians, av_medians, err_av_medians, beta_medians, err_beta_medians, mips_flux_medians, err_mips_flux_medians, halpha_snrs, hbeta_snrs, balmer_decs, err_balmer_decs_low, err_balmer_decs_high, balmer_avs, err_balmer_av_lows, err_balmer_av_highs, shapes_list, color_list, keys), columns=['axis_group','use_ratio_median', 'log_mass_median', variable+'_median', 'av_median', 'err_av_median', 'beta_median', 'err_beta_median', 'mips_flux_median', 'err_mips_flux_median', 'halpha_snr', 'hbeta_snr', 'balmer_dec', 'err_balmer_dec_low', 'err_balmer_dec_high', 'balmer_av', 'err_balmer_av_low', 'err_balmer_av_high', 'shape', 'color', 'key'])    
    if variable != 'log_use_ssfr':
        summary_df['log_use_ssfr_median'] = log_use_ssfr_medians
    if variable != 'log_use_sfr':
        summary_df['log_use_sfr_median'] = log_use_sfr_medians
    summary_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv', index=False)


    ax.set_xlabel('log(Stellar Mass)', fontsize=14) 
    ax.set_ylabel(variable, fontsize=14)
    ax.tick_params(labelsize=12)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Balmer Decrement', fontsize=14)
    ax.set_aspect(ellipse_width/ellipse_height)
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/sample_cut.pdf')


def plot_moved(n_groups=18, save_name='halpha_norm'):
    fig, ax = plt.subplots(figsize=(8,8))
    for axis_group in range(n_groups):
        ar_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_group_dfs/{axis_group}_df.csv').to_pandas()
        for i in range(len(ar_df)):
            row = ar_df.iloc[i]
            logssfr_old = np.log10(row['sfr2']/10**row['log_mass'])
            logssfr_new = row['log_ssfr']
            logmass = row['log_mass']
            ax.plot(logmass, logssfr_old, color='grey', marker='o')
            ax.plot(logmass, logssfr_new, color='red', marker='o')
            ax.plot([logmass, logmass], [logssfr_old, logssfr_new], color='blue', marker='None', ls='-')
    
    xlims = (9.0, 11.0)
    ylims = (-9.7, -8)
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)

    mass_width = 0.8
    split_width = 0.5
    starting_points = [(9.3, -9.1), (9.3, -8.6), (10.1, -9.1), (10.1, -8.6), (9.3, -9.6), (10.1, -9.6)]

    for j in range(6):
        rect = patches.Rectangle((starting_points[j][0],  starting_points[j][1]), mass_width, split_width, linestyle='--', linewidth=1, edgecolor='black', facecolor='none')
        ax.add_patch(rect)

    ax.set_xlabel('log(Stellar Mass)', fontsize=14)
    ax.set_ylabel('log(ssfr)', fontsize=14)

    fig.savefig(imd.axis_output_dir + '/old_new_sfr_comparison.pdf')
    # plt.show()

def plot_balmer_dec(save_name, n_groups, split_by, y_var = 'balmer_dec', color_var='log_ssfr'):
    '''Makes the balmer decrement plots. Now can also do AV and Beta instead of balmer dec on the y-axis

    Parameters:
    save_name (str): Folder to pull data from and save to
    n_groups (int): Number of axis ratio groups
    split_by (str): Column name that was used for splitting into groups in y-axis, used for coloring
    y_var (str): What to plot on the y-axis - either "balmer_dec", "av", or "beta"
    color_var (str): Colorbar variable for the plots

    '''

    # Fontsizes
    axis_fontsize = 14
    default_size = 7
    larger_size = 12


    # Axis limits
    if color_var == 'log_halpha_ssfr' or color_var == 'eq_width_ha':
        ylims = {
            'balmer_dec': (2, 10),
            'av': (0.25, 1.1),
            'beta': (-1.9, -0.95),
            'metallicity': (8.2, 8.9),
            'mips_flux': (0, 0.0063),
            'log_use_sfr': (-0.1, 2.6)
        }
    else:
        ylims = {
            'balmer_dec': (2, 7),
            'av': (0.25, 1.1),
            'beta': (-1.9, -0.95),
            'metallicity': (8.2, 8.9),
            'mips_flux': (0, 0.0063),
            'log_use_sfr': (-0.1, 2.6)
        }
    
    # Read in summary df
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv').to_pandas()

    color_str = ''
    cmap = mpl.cm.inferno 
    if color_var=='eq_width_ha':
        # eq color map
        norm = mpl.colors.Normalize(vmin=100, vmax=500) 
    elif color_var=='log_halpha_ssfr' or color_var=='log_ssfr' or color_var=='log_use_ssfr':
        # ssfr color map
        norm = mpl.colors.Normalize(vmin=-9.3, vmax=-8.1) 
    elif color_var=='log_use_sfr':
        # ssfr color map
        norm = mpl.colors.Normalize(vmin=0, vmax=2.5) 
        color_str='_log_use_sfr_color'
    elif color_var=='metallicity':
        # metallicity color map
        norm = mpl.colors.Normalize(vmin=8.1, vmax=8.9) 
        color_str='_metal_color'
    elif color_var=='log_use_ssfr':
        # metallicity color map
        norm = mpl.colors.Normalize(vmin=8.1, vmax=8.9) 
        color_str='_log_use_ssfr_color'

 
    # Get the length of the y-axis
    y_axis_len = ylims[y_var][1] - ylims[y_var][0]

    # Figure 1 - all the balmer decs in axis ratio vs balmer dec space
    fig, axarr = plt.subplots(1, 2, figsize=(20,8))
    ax_low_mass = axarr[0]
    ax_high_mass = axarr[1]

    for i in range(len(summary_df)):
        row = summary_df.iloc[i]

        # Set up the colormap on ssfr
        rgba = cmap(norm(row[color_var+'_median']))

        # Split into mass groups
        if row['log_mass_median'] < 10:
            ax = ax_low_mass
        elif row['log_mass_median'] >= 10:
            ax = ax_high_mass


        if y_var == 'balmer_dec':
            x_cord = row['use_ratio_median']
            y_cord = row['balmer_dec']
            
            ellipse_width, ellipse_height = get_ellipse_shapes(1.1, y_axis_len, row['shape'])

            # Make the point obvoise if hbeta signal_noise is low
            if row['hbeta_snr'] < 3:
                rgba = 'skyblue'


            ax.errorbar(x_cord, y_cord, yerr=np.array(row['err_balmer_dec_low'], row['err_balmer_dec_high']), marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_ylabel('Balmer Decrement', fontsize=axis_fontsize)
        elif y_var == 'av':
            x_cord = row['use_ratio_median']
            y_cord = row['av_median']

            ellipse_width, ellipse_height = get_ellipse_shapes(1.1, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, yerr=row['err_av_median'], marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_ylabel('FAST AV', fontsize=axis_fontsize)
        elif y_var == 'beta':
            x_cord = row['use_ratio_median']
            y_cord = row['beta_median']

            ellipse_width, ellipse_height = get_ellipse_shapes(1.1, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, yerr=row['err_beta_median'], marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_ylabel('Betaphot', fontsize=axis_fontsize)
        elif y_var == 'metallicity':
            x_cord = row['use_ratio_median']
            y_cord = row['metallicity_median']

            ellipse_width, ellipse_height = get_ellipse_shapes(1.1, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, yerr=row['err_metallicity_median'], marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_ylabel('Metallicity', fontsize=axis_fontsize)
        elif y_var == 'mips_flux':
            x_cord = row['use_ratio_median']
            y_cord = row['mips_flux_median']

            ellipse_width, ellipse_height = get_ellipse_shapes(1.1, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, yerr=row['err_mips_flux_median'], marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_ylabel('MIPS Flux', fontsize=axis_fontsize)
        elif y_var == 'log_use_sfr':
            x_cord = row['use_ratio_median']
            y_cord = row['log_use_sfr_median']

            ellipse_width, ellipse_height = get_ellipse_shapes(1.1, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_ylabel('log_use_sfr', fontsize=axis_fontsize)
    
    for ax in axarr:
        ax.set_xlabel('Axis Ratio', fontsize=axis_fontsize) 
        ax.tick_params(labelsize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(ylims[y_var])
        if y_var == 'balmer_dec':
            ax.text(-0.07, 1.6, 'Edge-on', fontsize=14, zorder=100)
            ax.text(0.95, 1.6, 'Face-on', fontsize=14, zorder=100)
    
    ax_low_mass.set_title('log(Stellar Mass) < 10', fontsize=axis_fontsize)
    ax_high_mass.set_title('log(Stellar Mass) > 10', fontsize=axis_fontsize)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axarr)
    cbar.set_label(color_var, fontsize=axis_fontsize)
    
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{y_var}_vs_ar{color_str}.pdf', bbox_inches='tight')



    # Figure 2 - Decrement vs mass
    fig, ax = plt.subplots(figsize=(9,8))


    for i in range(len(summary_df)):
        row = summary_df.iloc[i]

        # Set up the colormap on ssfr
        rgba = cmap(norm(row[color_var+'_median']))


        if y_var == 'balmer_dec':
            x_cord = row['log_mass_median']
            y_cord = row['balmer_dec']
            
            # Make the point obvoise if hbeta signal_noise is low
            if row['hbeta_snr'] < 3:
                rgba = 'skyblue'

            ellipse_width, ellipse_height = get_ellipse_shapes(1.5, y_axis_len, row['shape'])
            
            ax.errorbar(x_cord, y_cord, yerr=np.array(row['err_balmer_dec_low'], row['err_balmer_dec_high']), marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_ylabel('Balmer Decrement', fontsize=axis_fontsize)
        elif y_var == 'av':
            x_cord = row['log_mass_median']
            y_cord = row['av_median']

            ellipse_width, ellipse_height = get_ellipse_shapes(1.5, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, yerr=row['err_av_median'], marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_ylabel('FAST AV', fontsize=axis_fontsize)
        elif y_var == 'beta':
            x_cord = row['log_mass_median']
            y_cord = row['beta_median']
            
            ellipse_width, ellipse_height = get_ellipse_shapes(1.5, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, yerr=row['err_beta_median'], marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_ylabel('Betaphot', fontsize=axis_fontsize)
        elif y_var == 'metallicity':
            x_cord = row['log_mass_median']
            y_cord = row['metallicity_median']
            
            ellipse_width, ellipse_height = get_ellipse_shapes(1.5, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, yerr=row['err_metallicity_median'], marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_ylabel('Metallcity', fontsize=axis_fontsize)
        elif y_var == 'mips_flux':
            x_cord = row['log_mass_median']
            y_cord = row['mips_flux_median']
            
            ellipse_width, ellipse_height = get_ellipse_shapes(1.5, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, yerr=row['err_mips_flux_median'], marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_ylabel('MIPS Flux', fontsize=axis_fontsize)
        elif y_var == 'log_use_sfr':
            x_cord = row['use_ratio_median']
            y_cord = row['log_use_sfr_median']

            ellipse_width, ellipse_height = get_ellipse_shapes(1.1, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_ylabel('log_use_sfr', fontsize=axis_fontsize)



    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label(color_var, fontsize=axis_fontsize)
    ax.set_xlabel('log(Stellar Mass)', fontsize=axis_fontsize) 
    
    ax.tick_params(labelsize=12)
    ax.set_xlim(9.25, 10.75)
    ax.set_ylim(ylims[y_var])
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{y_var}_vs_mass{color_str}.pdf')

def bootstrap_median(df):
    """Bootstrap an error on a median from a column of a pandas dataframe
    
    Parameters:
    df (pd.DataFrame): One column of the dataframe

    Returns
    err_median (float): Bootstrapped median uncertainty
    """
    df = df[df>-999]
    n_samples = 10000
    samples = [np.random.choice(df, size=len(df)) for i in range(n_samples)]
    medians = [np.median(sample) for sample in samples]
    err_median = np.std(medians)
    return err_median



# main(nbins, save_name, split_by, stack_type, only_plot=only_plot)

# stack_all_continuum(6, save_name='mass_2bin_median')  
# main(12, 'eq_width_4bin' ,'balmer_dec')
# main(12, 'ssfr_4bin' ,'ssfr')

# plot_sample_split(12, 'ssfr_4bin', variable='ssfr')
# main(12, 'halpha_ssfr_4bin', 'ssfr', use_ha_ssfr=1)

# plot_balmer_dec('eq_width_4bin', 12, y_var='balmer_dec', color_var='eq_width_ha')
# plot_balmer_dec('eq_width_4bin', 12, y_var='av', color_var='eq_width_ha')
# plot_balmer_dec('eq_width_4bin', 12, y_var='beta', color_var='eq_width_ha')

# plot_all_spec_with_cont(6, 'mass_2bin_median')
# for axis_group in range(6):
#     fit_emission(0, 'cluster_norm', constrain_O3=False, axis_group=axis_group, save_name='mass_2bin_median', scaled='False', run_name='False')

# setup_new_stack_dir('eq_width_4bin')
# plot_moved()
