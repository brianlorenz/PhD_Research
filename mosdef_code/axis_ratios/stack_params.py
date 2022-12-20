from inspect import stack
from perform_axis_stack import stack_all_and_plot_all


class stack_params:
    
    def __init__(self, mass_width, split_width, starting_points, ratio_bins, nbins, split_by, save_name, stack_type, sfms_bins, use_whitaker_sfms, use_z_dependent_sfms, bootstrap, only_plot, run_stack):
        '''
        
        Parameters:
        mass_width (float): Length of a mass bin in log stellar masses, e.g. 0.7
        split_width (float): Length of a bin on the y-axis (ssfr or equivalent width, typically)
        starting_points (list of tuples): Each tuple is a coordinate in (mass,y) space for where to put the lower-left of each bin. The above widths determine the enclosed points
        ratio_bins (list): List of where to divide the bins in axis ratios, e.g [0.4, 0.7] gives 3 bins from 0-0.4, 0.4-0.7, 0.7-1
        nbins (int): Total number of bins, could be computed as len(starting points)*(len(ratio_bins)+1)
        split_by (str): Column name for the y-axis, either some form of ssfr ('log_use_ssfr') or 'eq_width_ha'
        save_name (str): Name for the directory where all outputs will be saved
        stack_type (str): 'mean' or 'median' for how to stack the spectra
        sfms_bins (boolean): Set to True to use different bins from the star-forming main sequence instead of the above method
        bootstrap (int): Set to 0 to not bootstrap, or the number of boostrap samples to run to do so
        only_plot (boolean): Set to True to skip the stacking step and just re-run the plots
        run_stack (boolean): Set to True to include this group in the current run of the code

        '''
        
        self.mass_width = mass_width
        self.split_width = split_width
        self.starting_points = starting_points
        self.ratio_bins = ratio_bins
        self.nbins = nbins
        self.split_by = split_by
        self.save_name = save_name
        self.stack_type = stack_type
        self.sfms_bins = sfms_bins
        self.use_whitaker_sfms = use_whitaker_sfms # Need sfms_bins = True to use
        self.use_z_dependent_sfms = use_z_dependent_sfms # Need sfms_bins = True to use
        self.bootstrap = bootstrap
        self.only_plot = only_plot
        self.run_stack = run_stack
        


# def make_whitaker_sfms_boot100(run_stack = False, only_plot = True):
#     run_stack = run_stack
#     only_plot = only_plot
#     mass_width = 1.0
#     split_width = 0.75
#     starting_points = [(9, -8.85), (10, -8.85), (9, -9.6), (10, -9.6)]
#     ratio_bins = [0.55]
#     nbins = 8
#     split_by = 'log_use_sfr'
#     save_name = 'whitaker_sfms_boot100_area_norm'
#     stack_type = 'median'
#     sfms_bins = True
#     use_whitaker_sfms = True
#     use_z_dependent_sfms = False
#     bootstrap = 100
#     both_ssfrs_4bin_mean_params = stack_params(mass_width, split_width, starting_points, ratio_bins, nbins, split_by, save_name, stack_type, sfms_bins, use_whitaker_sfms, use_z_dependent_sfms, bootstrap, only_plot, run_stack)
#     return both_ssfrs_4bin_mean_params
# whitaker_sfms_boot100 = make_whitaker_sfms_boot100()

# def make_whitaker_sfms_boot100_zdep(run_stack = False, only_plot = False):
#     run_stack = run_stack
#     only_plot = only_plot
#     mass_width = 1.0
#     split_width = 0.75
#     starting_points = [(9, -8.85), (10, -8.85), (9, -9.6), (10, -9.6)]
#     ratio_bins = [0.55]
#     nbins = 8
#     split_by = 'log_use_sfr'
#     save_name = 'whitaker_sfms_boot100_area_norm_zdep'
#     stack_type = 'median'
#     sfms_bins = True
#     use_whitaker_sfms = True
#     use_z_dependent_sfms = True
#     bootstrap = 100
#     both_ssfrs_4bin_mean_params = stack_params(mass_width, split_width, starting_points, ratio_bins, nbins, split_by, save_name, stack_type, sfms_bins, use_whitaker_sfms, use_z_dependent_sfms, bootstrap, only_plot, run_stack)
#     return both_ssfrs_4bin_mean_params
# whitaker_sfms_boot100_zdep = make_whitaker_sfms_boot100_zdep()

def make_indiv_fit_norm(run_stack = True, only_plot = True):
    run_stack = run_stack
    only_plot = only_plot
    mass_width = 1.0
    split_width = 0.75
    starting_points = [(9, -8.85), (10, -8.85), (9, -9.6), (10, -9.6)]
    ratio_bins = [0.55]
    nbins = 8
    split_by = 'log_use_sfr'
    save_name = 'norm_1_sn5_filtered'
    stack_type = 'median'
    sfms_bins = True
    use_whitaker_sfms = True
    use_z_dependent_sfms = True
    bootstrap = -1
    both_ssfrs_4bin_mean_params = stack_params(mass_width, split_width, starting_points, ratio_bins, nbins, split_by, save_name, stack_type, sfms_bins, use_whitaker_sfms, use_z_dependent_sfms, bootstrap, only_plot, run_stack)
    return both_ssfrs_4bin_mean_params
indiv_fit_norm = make_indiv_fit_norm()

def make_indiv_fit_norm_noaxis(run_stack = False, only_plot = False):
    run_stack = run_stack
    only_plot = only_plot
    mass_width = 1.0
    split_width = 0.75
    starting_points = [(9, -8.85), (10, -8.85), (9, -9.6), (10, -9.6)]
    ratio_bins = []
    nbins = 4
    split_by = 'log_use_sfr'
    save_name = 'norm_1_sn5_filtered_noaxis'
    stack_type = 'median'
    sfms_bins = True
    use_whitaker_sfms = True
    use_z_dependent_sfms = True
    bootstrap = -1
    both_ssfrs_4bin_mean_params = stack_params(mass_width, split_width, starting_points, ratio_bins, nbins, split_by, save_name, stack_type, sfms_bins, use_whitaker_sfms, use_z_dependent_sfms, bootstrap, only_plot, run_stack)
    return both_ssfrs_4bin_mean_params
indiv_fit_norm_noaxis = make_indiv_fit_norm_noaxis()

def make_indiv_fit_norm_3groups(run_stack = False, only_plot = False):
    run_stack = run_stack
    only_plot = only_plot
    mass_width = 1.0
    split_width = 0.75
    starting_points = [(9, -8.85), (10, -8.85), (9, -9.6), (10, -9.6)]
    ratio_bins = [0.4, 0.7]
    nbins = 12
    split_by = 'log_use_sfr'
    save_name = 'norm_1_sn5_filtered_3groups'
    stack_type = 'median'
    sfms_bins = True
    use_whitaker_sfms = True
    use_z_dependent_sfms = True
    bootstrap = 100
    both_ssfrs_4bin_mean_params = stack_params(mass_width, split_width, starting_points, ratio_bins, nbins, split_by, save_name, stack_type, sfms_bins, use_whitaker_sfms, use_z_dependent_sfms, bootstrap, only_plot, run_stack)
    return both_ssfrs_4bin_mean_params
indiv_fit_norm_3groups = make_indiv_fit_norm_3groups()



def make_test(run_stack = False, only_plot = False):
    run_stack = run_stack
    only_plot = only_plot
    mass_width = 1.0
    split_width = 0.75
    starting_points = [(9, -8.85), (10, -8.85), (9, -9.6), (10, -9.6)]
    ratio_bins = [0.55]
    nbins = 8
    split_by = 'log_use_sfr'
    save_name = 'test'
    stack_type = 'median'
    sfms_bins = True
    use_whitaker_sfms = True
    use_z_dependent_sfms = True
    bootstrap = 10
    both_ssfrs_4bin_mean_params = stack_params(mass_width, split_width, starting_points, ratio_bins, nbins, split_by, save_name, stack_type, sfms_bins, use_whitaker_sfms, use_z_dependent_sfms, bootstrap, only_plot, run_stack)
    return both_ssfrs_4bin_mean_params
test_stack = make_test()




stack_all_and_plot_all(indiv_fit_norm)
stack_all_and_plot_all(indiv_fit_norm_noaxis)
stack_all_and_plot_all(indiv_fit_norm_3groups)
# stack_all_and_plot_all(indiv_fit_norm_zdep)
# stack_all_and_plot_all(test_stack)
