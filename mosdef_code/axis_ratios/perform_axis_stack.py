from ensurepip import bootstrap
from initialize_mosdef_dirs import check_and_make_dir
from stack_spectra import *
from fit_emission import fit_emission
from stack_continuum import stack_all_continuum, plot_all_spec_with_cont, scale_all_bootstrapped_conts
import matplotlib as mpl
from plot_overlaid_spectra import plot_overlaid_spectra
from axis_group_metallicities import add_metals_to_summary_df, plot_metals, measure_metals, plot_group_metals_compare, plot_mass_metal
from balmer_avs import plot_balmer_stellar_avs
import random
import time
import json
from plot_sample_split import plot_sample_split
from plot_balmer_dec import plot_balmer_dec
from overview_plot import plot_overview


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
    bootstrap = param_class.bootstrap
    print(f'Running stack {save_name}. Making just the plots: {only_plot}')
    time_start = time.time()
    if only_plot==False:
        setup_new_stack_dir(save_name, param_class)
        stack_axis_ratio(mass_width, split_width, starting_points, ratio_bins, save_name, split_by, stack_type, sfms_bins, bootstrap=bootstrap)
        stack_all_continuum(nbins, save_name=save_name)
        time_stack = time.time()
        print(f'All stacking took {time_stack-time_start}')
        plot_all_spec_with_cont(nbins, save_name) # This is where the normalized cont is saved
        if bootstrap > 0:
            scale_all_bootstrapped_conts(nbins, save_name, bootstrap, make_plot=False) # This makes the normalized cont for all of the bootstraps
        for axis_group in range(nbins):
            fit_emission(0, 'cluster_norm', constrain_O3=False, axis_group=axis_group, save_name=save_name, scaled='False', run_name='False')
            if bootstrap > 0:
                for bootstrap_num in range(bootstrap):
                    fit_emission(0, 'cluster_norm', constrain_O3=False, axis_group=axis_group, save_name=save_name, scaled='False', run_name='False', bootstrap_num=bootstrap_num)
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
    print('starting plot')
    plot_overview(nbins, save_name, ratio_bins, starting_points, mass_width, split_width, sfms_bins, split_by)
    time_end = time.time()
    print(f'Total program took {time_end-time_start}')

def setup_new_stack_dir(save_name, param_class):
    """Sets up the directory with all the necessary folders"""
    check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}')
    sub_dirs = ['composite_images', 'composite_seds', 'cont_subs', 'conts', 'emission_fits', 'emission_images', 'spectra', 'spectra_images', 'group_dfs']
    for name in sub_dirs:
        check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_{name}')
    check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts')    
    
    f = open(imd.axis_cluster_data_dir + f"/{save_name}/{save_name}_stack_params.txt", "w")
    param_str = json.dumps(param_class.__dict__)
    f.write(param_str)
    f.close()
    return



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
