from matplotlib.pyplot import axis
from stack_spectra import *
from fit_emission import fit_emission
import matplotlib as mpl


mass_width = 0.7
ssfr_width = 0.5
starting_points = [(9.3, -9.1), (9.3, -8.6), (10.0, -9.2), (10.0, -8.7)]
ratio_bins = [0.4, 0.7]


shapes = {'low': '^', 'mid': 'o', 'high': 'x'}
colors = {'lowm_lows': 'red', 'lowm_highs': 'blue', 'highm_lows': 'orange', 'highm_highs': 'mediumseagreen'}


def main():
    '''performs all the steps to get this group plotted'''
    stack_axis_ratio(3, mass_width, ssfr_width, starting_points, ratio_bins)
    for axis_group in range(12):
        fit_emission(0, 'cluster_norm', constrain_O3=False, axis_group=axis_group, save_name='mass_ssfr', scaled='False', run_name='False')
    plot_sample_split()




def plot_sample_split(n_groups = 12, save_name = 'mass_ssfr'):
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

    # Figure 1 - Showing how the sample gets cut
    # Plot grey points for galaxies not included
    all_axis_ratio_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()
    all_axis_ratio_df['log_ssfr'] = np.log10((10**all_axis_ratio_df['log_sfr']) / (10**all_axis_ratio_df['log_mass']))

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
    ax.plot(all_axis_ratio_df['log_mass'], all_axis_ratio_df['log_ssfr'], color = 'grey', ls='None', marker='o')

    for axis_group in range(n_groups):
        axis_ratio_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_group_dfs/{axis_group}_df.csv').to_pandas()

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
                color = colors["lowm_lows"]
        if mass_median > starting_points[1][0] and mass_median < starting_points[1][0] + mass_width:
            if ssfr_median > starting_points[1][1] and ssfr_median < starting_points[1][1] + ssfr_width:
                color = colors["lowm_highs"]
        if mass_median > starting_points[2][0] and mass_median < starting_points[2][0] + mass_width:
            if ssfr_median > starting_points[2][1] and ssfr_median < starting_points[2][1] + ssfr_width:
                color = colors["highm_lows"]
        if mass_median > starting_points[3][0] and mass_median < starting_points[3][0] + mass_width:
            if ssfr_median > starting_points[3][1] and ssfr_median < starting_points[3][1] + ssfr_width:
                color = colors["highm_highs"]

        
        group_num.append(axis_group)
        shapes_list.append(shape)
        color_list.append(color)
        axis_medians.append(axis_median)
        mass_medians.append(mass_median)
        ssfr_medians.append(ssfr_median)

        # plot a black point at the median
        ax.plot(mass_median, ssfr_median, ls='None', marker=shape, color='black', zorder=100)

        # Plot the rest of the points
        ax.plot(axis_ratio_df['log_mass'], axis_ratio_df['log_ssfr'], color = color, ls='None', marker=shape)



    summary_df = pd.DataFrame(zip(group_num, axis_medians, mass_medians, ssfr_medians, shapes_list, color_list), columns=['axis_group','use_ratio_median', 'log_mass_median', 'log_ssfr_median', 'shape', 'color'])
    summary_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv', index=False)


    ax.set_xlabel('log(Stellar Mass)', fontsize=14) 
    ax.set_ylabel('log(ssfr)', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_ylim(-9.7, -8)
    ax.set_xlim(9.0, 11.0)
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/sample_cut.pdf')


def plot_balmer_dec(save_name):
    '''Makes the balmer decrement plots'''
    
    summary_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/summary.csv').to_pandas()

    axis_groups = []
    balmer_decs = []
    balmer_err_lows = []
    balmer_err_highs = []

    for axis_group in range(12):
        emission_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_fits/{axis_group}_emission_fits.csv').to_pandas()
        axis_groups.append(axis_group)
        balmer_decs.append(emission_df.iloc[0]['balmer_dec'])
        balmer_err_lows.append(emission_df.iloc[0]['err_balmer_dec_low'])
        balmer_err_highs.append(emission_df.iloc[0]['err_balmer_dec_high'])

    balmer_df = pd.DataFrame(zip(axis_groups, balmer_decs, balmer_err_lows, balmer_err_highs), columns=['axis_group', 'balmer_dec', 'err_balmer_dec_low', 'err_balmer_dec_high'])

    summary_df = summary_df.merge(balmer_df, left_on='axis_group', right_on='axis_group')

    # Figure 1 - all the balmer decs in axis ratio vs balmer dec space
    fig, ax = plt.subplots(figsize=(8,8))

    for i in range(len(summary_df)):
        row = summary_df.iloc[i]
        ax.errorbar(row['use_ratio_median'], row['balmer_dec'], yerr=np.array(row['err_balmer_dec_low'], row['err_balmer_dec_high']), marker=row['shape'], color=row['color'])


    ax.set_xlabel('Axis Ratio', fontsize=14) 
    ax.set_ylabel('Balmer Decrement', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_ylim(0, 6)
    ax.set_xlim(-0.05, 1.05)
    ax.text(-0.07, -0.6, 'Edge-on', fontsize=14, zorder=100)
    ax.text(0.95, -0.6, 'Face-on', fontsize=14, zorder=100)
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/balmer_vs_ar.pdf')


    # Figure 2 - Decrement vs mass
    fig, ax = plt.subplots(figsize=(9,8))

    cmap = mpl.cm.viridis 
    norm = mpl.colors.Normalize(vmin=-9.4, vmax=-8) 

    for i in range(len(summary_df)):
        row = summary_df.iloc[i]
        rgba = cmap(norm(row['log_ssfr_median']))
        ax.errorbar(row['log_mass_median'], row['balmer_dec'], yerr=np.array(row['err_balmer_dec_low'], row['err_balmer_dec_high']), marker=row['shape'], color=rgba)

    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)
    cbar.set_label('log(ssfr)', fontsize=14)
    ax.set_xlabel('log(Stellar Mass)', fontsize=14) 
    ax.set_ylabel('Balmer Decrement', fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_ylim(0, 6)
    ax.set_xlim(9.25, 10.75)
    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/balmer_vs_mass.pdf')


    

plot_sample_split()