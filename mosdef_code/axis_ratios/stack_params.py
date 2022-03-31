from inspect import stack
from perform_axis_stack import stack_all_and_plot_all


class stack_params:
    
    def __init__(self, mass_width, split_width, starting_points, ratio_bins, nbins, split_by, save_name, stack_type, sfms_bins, only_plot, run_stack):
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
        self.only_plot = only_plot
        self.run_stack = run_stack

# Equalivent width ha
def make_eq_width_ha_params(run_stack = False, only_plot = False):
    run_stack = run_stack
    only_plot = only_plot
    mass_width = 0.8
    split_width = 300
    starting_points = [(9.3, 0), (10.1, 0), (9.3, 300), (10.1, 300)]
    ratio_bins = [0.4, 0.7]
    nbins = 12
    split_by = 'eq_width_ha'
    save_name = 'eq_width_4bin_mean'
    stack_type = 'mean'
    sfms_bins = False
    eq_width_ha_params = stack_params(mass_width, split_width, starting_points, ratio_bins, nbins, split_by, save_name, stack_type, sfms_bins, only_plot, run_stack)
    return eq_width_ha_params
eq_width_ha_params = make_eq_width_ha_params()

# 2 axis bins, combined ssfrs
def make_both_ssfrs_4bin_mean_2axis_params(run_stack = False, only_plot = False):
    run_stack = run_stack
    only_plot = only_plot
    mass_width = 0.8
    split_width = 0.75
    starting_points = [(9.3, -8.85), (10.1, -8.85), (9.3, -9.6), (10.1, -9.6)]
    ratio_bins = [0.55]
    nbins = 8
    split_by = 'log_use_ssfr'
    save_name = 'both_ssfrs_4bin_mean-2axis'
    stack_type = 'mean'
    sfms_bins = False
    both_ssfrs_4bin_mean_2axis_params = stack_params(mass_width, split_width, starting_points, ratio_bins, nbins, split_by, save_name, stack_type, sfms_bins, only_plot, run_stack)
    return both_ssfrs_4bin_mean_2axis_params
both_ssfrs_4bin_mean_2axis_params = make_both_ssfrs_4bin_mean_2axis_params()

# Normal 12bin using combined sfrs, using sfr2 when both lines are good, halpha_sfr when hbeta is below 3 sigma (or not covered)
def make_both_ssfrs_4bin_mean_params(run_stack = False, only_plot = False):
    run_stack = run_stack
    only_plot = only_plot
    mass_width = 0.8
    split_width = 0.75
    starting_points = [(9.3, -8.85), (10.1, -8.85), (9.3, -9.6), (10.1, -9.6)]
    ratio_bins = [0.4, 0.7]
    nbins = 12
    split_by = 'log_use_ssfr'
    save_name = 'both_ssfrs_4bin_mean'
    stack_type = 'mean'
    sfms_bins = False
    both_ssfrs_4bin_mean_params = stack_params(mass_width, split_width, starting_points, ratio_bins, nbins, split_by, save_name, stack_type, sfms_bins, only_plot, run_stack)
    return both_ssfrs_4bin_mean_params
both_ssfrs_4bin_mean_params = make_both_ssfrs_4bin_mean_params()

# Normal 12bin using combined sfrs, using sfr2 when both lines are good, halpha_sfr when hbeta is below 3 sigma (or not covered)
def make_both_ssfrs_4bin_median_params(run_stack = False, only_plot = True):
    run_stack = run_stack
    only_plot = only_plot
    mass_width = 0.8
    split_width = 0.75
    starting_points = [(9.3, -8.85), (10.1, -8.85), (9.3, -9.6), (10.1, -9.6)]
    ratio_bins = [0.4, 0.7]
    nbins = 12
    split_by = 'log_use_ssfr'
    save_name = 'both_ssfrs_4bin_median'
    stack_type = 'median'
    sfms_bins = False
    both_ssfrs_4bin_mean_params = stack_params(mass_width, split_width, starting_points, ratio_bins, nbins, split_by, save_name, stack_type, sfms_bins, only_plot, run_stack)
    return both_ssfrs_4bin_mean_params
both_ssfrs_4bin_median_params = make_both_ssfrs_4bin_median_params()

# Normal 12bin using combined sfrs, using sfr2 when both lines are good, halpha_sfr when hbeta is below 3 sigma (or not covered)
def make_both_ssfrs_4bin_2axis_median_params(run_stack = False, only_plot = True):
    run_stack = run_stack
    only_plot = only_plot
    mass_width = 0.8
    split_width = 0.75
    starting_points = [(9.3, -8.85), (10.1, -8.85), (9.3, -9.6), (10.1, -9.6)]
    ratio_bins = [0.55]
    nbins = 8
    split_by = 'log_use_ssfr'
    save_name = 'both_ssfrs_4bin_median_2axis'
    stack_type = 'median'
    sfms_bins = False
    both_ssfrs_4bin_mean_params = stack_params(mass_width, split_width, starting_points, ratio_bins, nbins, split_by, save_name, stack_type, sfms_bins, only_plot, run_stack)
    return both_ssfrs_4bin_mean_params
both_ssfrs_4bin_2axis_median_params = make_both_ssfrs_4bin_2axis_median_params()

# Normal 12bin using combined sfrs, using sfr2 when both lines are good, halpha_sfr when hbeta is below 3 sigma (or not covered)
def make_both_sfms_6bin_2axis_median_params(run_stack = True, only_plot = False):
    run_stack = run_stack
    only_plot = only_plot
    mass_width = 0.8
    split_width = 0.75
    starting_points = [(9.3, -8.85), (10.1, -8.85), (9.3, -9.6), (10.1, -9.6)]
    ratio_bins = [0.55]
    nbins = 12
    split_by = 'log_use_sfr'
    save_name = 'both_sfms_6bin_median_2axis'
    stack_type = 'median'
    sfms_bins = True
    both_ssfrs_4bin_mean_params = stack_params(mass_width, split_width, starting_points, ratio_bins, nbins, split_by, save_name, stack_type, sfms_bins, only_plot, run_stack)
    return both_ssfrs_4bin_mean_params
both_sfms_6bin_2axis_median_params = make_both_sfms_6bin_2axis_median_params()

# Normal 12bin using only sfr2 rates + lower limits
def make_mosdef_ssfr_6bin_mean_params(run_stack = False, only_plot = False):
    run_stack = run_stack
    only_plot = only_plot
    mass_width = 0.8
    split_width = 0.75
    starting_points = [(9.3, -8.85), (10.1, -8.85), (9.3, -9.6), (10.1, -9.6)]
    ratio_bins = [0.4, 0.7]
    nbins = 12
    split_by = 'log_ssfr'
    save_name = 'mosdef_ssfr_4bin_mean'
    stack_type = 'mean'
    sfms_bins = False
    mosdef_ssfr_4bin_mean_params = stack_params(mass_width, split_width, starting_points, ratio_bins, nbins, split_by, save_name, stack_type, sfms_bins, only_plot, run_stack)
    return mosdef_ssfr_4bin_mean_params
mosdef_ssfr_6bin_mean_params = make_mosdef_ssfr_6bin_mean_params()

# Normal 12bin using only sfr2 rates + lower limits, with median stack
def make_mosdef_ssfr_4bin_median_params(run_stack = False, only_plot = False):
    run_stack = run_stack
    only_plot = only_plot
    mass_width = 0.8
    split_width = 0.75
    starting_points = [(9.3, -8.85), (10.1, -8.85), (9.3, -9.6), (10.1, -9.6)]
    ratio_bins = [0.4, 0.7]
    nbins = 12
    split_by = 'log_ssfr'
    save_name = 'mosdef_ssfr_4bin_median'
    stack_type = 'median'
    sfms_bins = False
    mosdef_ssfr_4bin_median_params = stack_params(mass_width, split_width, starting_points, ratio_bins, nbins, split_by, save_name, stack_type, sfms_bins, only_plot, run_stack)
    return mosdef_ssfr_4bin_median_params
mosdef_ssfr_4bin_median_params = make_mosdef_ssfr_4bin_median_params()

# 12 bins halpha sfrs only,shifted the left boxes higher
def make_halpha_ssfr_4bin_mean_shifted_params(run_stack = False, only_plot = False):
    run_stack = run_stack
    only_plot = only_plot
    mass_width = 0.8
    split_width = 0.75
    starting_points = [(9.3, -8.55), (10.1, -8.85), (9.3, -9.3), (10.1, -9.6)]
    ratio_bins = [0.4, 0.7]
    nbins = 12
    split_by = 'log_halpha_ssfr'
    save_name = 'halpha_ssfr_4bin_mean_shifted'
    stack_type = 'mean'
    sfms_bins = False
    halpha_ssfr_4bin_mean_shifted_params = stack_params(mass_width, split_width, starting_points, ratio_bins, nbins, split_by, save_name, stack_type, sfms_bins, only_plot, run_stack)
    return halpha_ssfr_4bin_mean_shifted_params
halpha_ssfr_4bin_mean_shifted_params = make_halpha_ssfr_4bin_mean_shifted_params()

# 12 bins, 2 mass 2 ssfr, new halpha ssfrs
# run_stack = True
# only_plot = False
# mass_width = 0.8
# split_width = 0.75
# starting_points = [(9.3, -8.85), (10.1, -8.85), (9.3, -9.6), (10.1, -9.6)]
# ratio_bins = [0.4, 0.7]
# nbins = 12
# split_by = 'log_halpha_ssfr'
# save_name = 'halpha_ssfr_4bin_mean'
# stack_type = 'mean'


# stack_all_and_plot_all(eq_width_ha_params)
# stack_all_and_plot_all(both_ssfrs_4bin_mean_2axis_params)
# stack_all_and_plot_all(both_ssfrs_4bin_mean_params)
stack_all_and_plot_all(both_ssfrs_4bin_median_params)
stack_all_and_plot_all(both_ssfrs_4bin_2axis_median_params)
stack_all_and_plot_all(both_sfms_6bin_2axis_median_params)
# stack_all_and_plot_all(mosdef_ssfr_4bin_mean_params)
# stack_all_and_plot_all(mosdef_ssfr_4bin_median_params)
# stack_all_and_plot_all(halpha_ssfr_4bin_mean_shifted_params)
