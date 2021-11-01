from matplotlib.pyplot import axis
from stack_spectra import *
from fit_emission import fit_emission
import matplotlib as mpl
from matplotlib.patches import Ellipse
from ellipses_for_plotting import get_ellipse_shapes

mass_width = 0.8
ssfr_width = 0.5
starting_points = [(9.3, -9.1), (9.3, -8.6), (10.1, -9.1), (10.1, -8.6), (9.3, -9.6), (10.1, -9.6)]
ratio_bins = [0.4, 0.7]


shapes = {'low': '+', 'mid': 'd', 'high': 'o'}
colors = {'lowm_lows': 'red', 'lowm_highs': 'blue', 'highm_lows': 'orange', 'highm_highs': 'mediumseagreen', 'lowest_lows': 'lightskyblue', 'lowest_highs': 'darkviolet'}


def main():
    '''performs all the steps to get this group plotted'''
    stack_axis_ratio(3, mass_width, ssfr_width, starting_points, ratio_bins)
    for axis_group in range(18):
        fit_emission(0, 'cluster_norm', constrain_O3=False, axis_group=axis_group, save_name='halpha_norm', scaled='False', run_name='False')



def plot_sample_split(n_groups = 18, save_name = 'halpha_norm'):
    """Plots the way that the sample has been divided in mass/ssfr space"""

    fig, ax = plt.subplots(figsize=(8,8))
    
    # Check to make sure lines were used, same coverage as when generating the spectrum
    coverage_list = [
                ('Halpha', 6564.61),
                ('Hbeta', 4862.68)
            ]
    
    
    group_num = []
    shapes_list = []
    color_list = []
    axis_medians = []
    mass_medians = []
    ssfr_medians = []
    av_medians = []
    beta_medians = []
    keys = []

    # Figure 1 - Showing how the sample gets cut
    # Plot grey points for galaxies not included
    all_axis_ratio_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()
    all_axis_ratio_df['log_ssfr'] = np.log10((all_axis_ratio_df['sfr']) / (10**all_axis_ratio_df['log_mass']))

    # Check if each galaxy was included - if not, drop it from the dataframe
    drop_idx = []
    for i in range(len(all_axis_ratio_df)):
        mosdef_obj = get_mosdef_obj(all_axis_ratio_df.iloc[i]['field'], all_axis_ratio_df.iloc[i]['v4id'])
        
        covered = check_line_coverage(mosdef_obj, coverage_list)
        if covered == False:
            drop_idx.append(False)
        else:
            drop_idx.append(True)
    all_axis_ratio_df = all_axis_ratio_df[drop_idx]
    ax.plot(all_axis_ratio_df['log_mass'], all_axis_ratio_df['log_ssfr'], color = 'grey', ls='None', marker='o', zorder=1)

    cmap = mpl.cm.viridis 
    norm = mpl.colors.Normalize(vmin=2, vmax=7) 

    for axis_group in range(n_groups):
        axis_ratio_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_group_dfs/{axis_group}_df.csv').to_pandas()

        axis_ratio_df['balmer_dec'] = (axis_ratio_df['ha_flux'] / axis_ratio_df['hb_flux'])


        # Check if each galaxy was included - if not, drop it from the dataframe
        drop_idx = []
        for i in range(len(axis_ratio_df)):
            mosdef_obj = get_mosdef_obj(axis_ratio_df.iloc[i]['field'], axis_ratio_df.iloc[i]['v4id'])
            
            covered = check_line_coverage(mosdef_obj, coverage_list)
            if covered == False:
                drop_idx.append(False)
            else:
                drop_idx.append(True)
        axis_ratio_df = axis_ratio_df[drop_idx]


        # Determine what color and shape to plot with
        axis_median = np.median(axis_ratio_df['use_ratio'])
        mass_median = np.median(axis_ratio_df['log_mass'])
        ssfr_median = np.median(axis_ratio_df['log_ssfr'])
        av_median = np.median(axis_ratio_df['AV'])
        beta_median = np.median(axis_ratio_df['beta'])

        # Figure out what axis ratio bin
        if axis_median < ratio_bins[0]:
            shape = shapes['low']
        elif axis_median > ratio_bins[1]:
            shape = shapes['high']
        else:
            shape = shapes['mid']

        # Figure out which mass/ssfr bin
        color = 'purple'
        if mass_median > starting_points[0][0] and mass_median < starting_points[0][0] + mass_width:
            if ssfr_median > starting_points[0][1] and ssfr_median < starting_points[0][1] + ssfr_width:
                key = "lowm_lows"
                # Create a Rectangle patch
                rect = patches.Rectangle((starting_points[0][0],  starting_points[0][1]), mass_width, ssfr_width, linestyle='--', linewidth=1, edgecolor=colors[key], facecolor='none')
        if mass_median > starting_points[1][0] and mass_median < starting_points[1][0] + mass_width:
            if ssfr_median > starting_points[1][1] and ssfr_median < starting_points[1][1] + ssfr_width:
                key = "lowm_highs"
                rect = patches.Rectangle((starting_points[1][0],  starting_points[1][1]), mass_width, ssfr_width, linestyle='--', linewidth=1, edgecolor=colors[key], facecolor='none')
        if mass_median > starting_points[2][0] and mass_median < starting_points[2][0] + mass_width:
            if ssfr_median > starting_points[2][1] and ssfr_median < starting_points[2][1] + ssfr_width:
                key = "highm_lows"
                rect = patches.Rectangle((starting_points[2][0],  starting_points[2][1]), mass_width, ssfr_width, linestyle='--', linewidth=1, edgecolor=colors[key], facecolor='none')
        if mass_median > starting_points[3][0] and mass_median < starting_points[3][0] + mass_width:
            if ssfr_median > starting_points[3][1] and ssfr_median < starting_points[3][1] + ssfr_width:
                key = "highm_highs"
                rect = patches.Rectangle((starting_points[3][0],  starting_points[3][1]), mass_width, ssfr_width, linestyle='--', linewidth=1, edgecolor=colors[key], facecolor='none')
        if mass_median > starting_points[4][0] and mass_median < starting_points[4][0] + mass_width:
            if ssfr_median > starting_points[4][1] and ssfr_median < starting_points[4][1] + ssfr_width:
                key = "lowest_lows"
                rect = patches.Rectangle((starting_points[4][0],  starting_points[4][1]), mass_width, ssfr_width, linestyle='--', linewidth=1, edgecolor=colors[key], facecolor='none')
        if mass_median > starting_points[5][0] and mass_median < starting_points[5][0] + mass_width:
            if ssfr_median > starting_points[5][1] and ssfr_median < starting_points[5][1] + ssfr_width:
                key = "lowest_highs"
                rect = patches.Rectangle((starting_points[5][0],  starting_points[5][1]), mass_width, ssfr_width, linestyle='--', linewidth=1, edgecolor=colors[key], facecolor='none')
        color = colors[key]

        
        group_num.append(axis_group)
        shapes_list.append(shape)
        color_list.append(color)
        axis_medians.append(axis_median)
        mass_medians.append(mass_median)
        ssfr_medians.append(ssfr_median)
        av_medians.append(av_median)
        beta_medians.append(beta_median)
        keys.append(key)



        # Set the axis limits
        xlims = (9.0, 11.0)
        ylims = (-9.7, -8)
        ax.set_ylim(ylims)
        ax.set_xlim(xlims)

        # Get the ellipse shapes for plotting
        ellipse_width, ellipse_height = get_ellipse_shapes(xlims[1]-xlims[0], np.abs(ylims[1]-ylims[0]), shape)

        # plot a black point at the median
        # ax.plot(mass_median, ssfr_median, ls='None', marker=shape, color='black', zorder=100)
        ax.add_artist(Ellipse((mass_median, ssfr_median), ellipse_width, ellipse_height, facecolor='black', zorder=3))

        


        for i in range(len(axis_ratio_df)):
            row = axis_ratio_df.iloc[i]
            rgba = cmap(norm(row['balmer_dec']))
            # ax.plot(row['log_mass'], row['log_ssfr'], color = rgba, ls='None', marker=shape)
        
            ax.add_artist(Ellipse((row['log_mass'], row['log_ssfr']), ellipse_width, ellipse_height, facecolor=rgba, zorder=2))

        # Plot the rest of the points
        # ax.plot(axis_ratio_df['log_mass'], axis_ratio_df['log_ssfr'], color = color, ls='None', marker=shape)

        # Add the patch to the Axes
        ax.add_patch(rect)



    summary_df = pd.DataFrame(zip(group_num, axis_medians, mass_medians, ssfr_medians, av_medians, beta_medians, shapes_list, color_list, keys), columns=['axis_group','use_ratio_median', 'log_mass_median', 'log_ssfr_median', 'av_median', 'beta_median', 'shape', 'color', 'key'])
    summary_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv', index=False)


    ax.set_xlabel('log(Stellar Mass)', fontsize=14) 
    ax.set_ylabel('log(ssfr)', fontsize=14)
    ax.tick_params(labelsize=12)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('Balmer Decrement', fontsize=14)
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/sample_cut.pdf')


def plot_balmer_dec(save_name, n_groups, y_var = 'balmer_dec'):
    '''Makes the balmer decrement plots. Now can also do AV and Beta instead of balmer dec on the y-axis

    Parameters:
    save_name (str): Folder to pull data from and save to
    n_groups (int): Number of axis ratio groups
    y_var (str): What to plot on the y-axis - either "balmer_dec", "av", or "beta"

    '''

    # Fontsizes
    axis_fontsize = 14
    default_size = 7
    larger_size = 12

    # ssfr color map
    cmap = mpl.cm.inferno 
    norm = mpl.colors.Normalize(vmin=-9.3, vmax=-8.1) 


    # Axis limits
    ylims = {
        'balmer_dec': (2, 7),
        'av': (0.25, 1.1),
        'beta': (-1.75, -1.05)
    }
    
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv').to_pandas()

    axis_groups = []
    balmer_decs = []
    balmer_err_lows = []
    balmer_err_highs = []

    for axis_group in range(n_groups):
        emission_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_fits/{axis_group}_emission_fits.csv').to_pandas()
        axis_groups.append(axis_group)
        balmer_decs.append(emission_df.iloc[0]['balmer_dec'])
        balmer_err_lows.append(emission_df.iloc[0]['err_balmer_dec_low'])
        balmer_err_highs.append(emission_df.iloc[0]['err_balmer_dec_high'])

    balmer_df = pd.DataFrame(zip(axis_groups, balmer_decs, balmer_err_lows, balmer_err_highs), columns=['axis_group', 'balmer_dec', 'err_balmer_dec_low', 'err_balmer_dec_high'])

    summary_df = summary_df.merge(balmer_df, left_on='axis_group', right_on='axis_group')

    # Get the length of the y-axis
    y_axis_len = ylims[y_var][1] - ylims[y_var][0]

    # Figure 1 - all the balmer decs in axis ratio vs balmer dec space
    fig, axarr = plt.subplots(1, 2, figsize=(20,8))
    ax_low_mass = axarr[0]
    ax_high_mass = axarr[1]

    for i in range(len(summary_df)):
        row = summary_df.iloc[i]

        # Set up the colormap on ssfr
        rgba = cmap(norm(row['log_ssfr_median']))

        # Split into mass groups
        if row['log_mass_median'] < 10:
            ax = ax_low_mass
        elif row['log_mass_median'] >= 10:
            ax = ax_high_mass


        if y_var == 'balmer_dec':
            x_cord = row['use_ratio_median']
            y_cord = row['balmer_dec']
            
            ellipse_width, ellipse_height = get_ellipse_shapes(1.1, y_axis_len, row['shape'])

            ax.errorbar(x_cord, y_cord, yerr=np.array(row['err_balmer_dec_low'], row['err_balmer_dec_high']), marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_ylabel('Balmer Decrement', fontsize=axis_fontsize)
        elif y_var == 'av':
            ellipse_width, ellipse_height = get_ellipse_shapes(1.1, y_axis_len, row['shape'])

            ax.add_artist(Ellipse((row['use_ratio_median'], row['av_median']), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_ylabel('FAST AV', fontsize=axis_fontsize)
        elif y_var == 'beta':
            ellipse_width, ellipse_height = get_ellipse_shapes(1.1, y_axis_len, row['shape'])

            ax.add_artist(Ellipse((row['use_ratio_median'], row['beta_median']), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_ylabel('Betaphot', fontsize=axis_fontsize)
    
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
    cbar.set_label('log(ssfr)', fontsize=axis_fontsize)
    
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{y_var}_vs_ar.pdf', bbox_inches='tight')



    # Figure 2 - Decrement vs mass
    fig, ax = plt.subplots(figsize=(9,8))


    for i in range(len(summary_df)):
        row = summary_df.iloc[i]

        # Set up the colormap on ssfr
        rgba = cmap(norm(row['log_ssfr_median']))


        if y_var == 'balmer_dec':
            x_cord = row['log_mass_median']
            y_cord = row['balmer_dec']
            
            ellipse_width, ellipse_height = get_ellipse_shapes(1.5, y_axis_len, row['shape'])
            
            ax.errorbar(x_cord, y_cord, yerr=np.array(row['err_balmer_dec_low'], row['err_balmer_dec_high']), marker='None', color=rgba)
            ax.add_artist(Ellipse((x_cord, y_cord), ellipse_width, ellipse_height, facecolor=rgba))
        elif y_var == 'av':
            ellipse_width, ellipse_height = get_ellipse_shapes(1.5, y_axis_len, row['shape'])

            ax.add_artist(Ellipse((row['log_mass_median'], row['av_median']), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_ylabel('FAST AV', fontsize=axis_fontsize)
        elif y_var == 'beta':
            ellipse_width, ellipse_height = get_ellipse_shapes(1.5, y_axis_len, row['shape'])
            ax.add_artist(Ellipse((row['log_mass_median'], row['beta_median']), ellipse_width, ellipse_height, facecolor=rgba))
            ax.set_ylabel('Betaphot', fontsize=axis_fontsize)



    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('log(ssfr)', fontsize=axis_fontsize)
    ax.set_xlabel('log(Stellar Mass)', fontsize=axis_fontsize) 
    
    ax.tick_params(labelsize=12)
    ax.set_xlim(9.25, 10.75)
    ax.set_ylim(ylims[y_var])
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{y_var}_vs_mass.pdf')


# stack_axis_ratio(3, mass_width, ssfr_width, starting_points, ratio_bins)
    
# main()
plot_sample_split()
plot_balmer_dec('halpha_norm', 18, y_var='balmer_dec')
plot_balmer_dec('halpha_norm', 18, y_var='av')
plot_balmer_dec('halpha_norm', 18, y_var='beta')