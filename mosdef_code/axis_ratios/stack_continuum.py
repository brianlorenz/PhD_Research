import glob
import os
import initialize_mosdef_dirs as imd
from astropy.io import ascii
from scipy import interpolate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stack_spectra import plot_spec
from cross_correlate import get_cross_cor


line_regions = [
    (6300, 6900),
    (4600, 5200)
]

mask_regions = [
    (4800, 4900),
    (4920, 5040),
    (6500, 6600)
]

# Run in the order - stack, plot


def stack_continuum(axis_group, save_name):
    '''Stacks all of the normalized continuum fast fits'''
    cont_folder = imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/{axis_group}_conts/'
    file_names = glob.glob(os.path.join(cont_folder, '*'))

    cont_dfs = [ascii.read(file).to_pandas() for file in file_names]
    interp_conts = [interpolate.interp1d(cont_df['rest_wavelength'], cont_df['f_lambda_norm'], fill_value=0, bounds_error=False) for cont_df in cont_dfs]
    
    spec_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_spectra/{axis_group}_spectrum.csv').to_pandas()

    spectrum_wavelength = spec_df['wavelength']
    interp_fluxes = [interp_cont(spectrum_wavelength) for interp_cont in interp_conts]
    summed_flux = np.mean(interp_fluxes, axis=0)         
    sum_cont_df = pd.DataFrame(zip(spectrum_wavelength, summed_flux), columns = ['rest_wavelength', 'f_lambda'])
    sum_cont_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts/{axis_group}_summed_cont.csv', index=False)

def stack_all_continuum(n_clusters, save_name):
    for axis_group in range(n_clusters):
        print(f'Stacking group {axis_group}')
        stack_continuum(axis_group, save_name)

def scale_bootstrapped_conts(axis_group, save_name, bootstrap=0, make_plot=False):
    '''Scales the continuums for each of the bootstrapped spectra
    
    NOTE - still only uses the summed cont from above, probably should re-sum the continuum for each bootstrap
    
    Parameters:
    axis_group (int): Group number
    save_name (str): Name of the location to save
    bootstrap (int): Number of bootstrapped samples
    make_plot (boolean): Set to True to make and save plots for each spectrum
    '''
    sum_cont_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts/{axis_group}_summed_cont.csv').to_pandas()
    imd.check_and_make_dir(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts_boots/')
    for bootstrap_num in range(bootstrap):
        plot_spec_with_cont(axis_group, save_name, bootstrap_num=bootstrap_num, sum_cont_df=sum_cont_df, use_sum_cont_df=True, make_plot=make_plot)

def scale_all_bootstrapped_conts(n_clusters, save_name, bootstrap=0, make_plot=False):
    '''Scales the continuums for each of the bootstrapped spectra
    
    NOTE - still only uses the summed cont from above, probably should re-sum the continuum for each bootstrap
    
    Parameters:
    n_clusters (int): Number of groups
    save_name (str): Name of the location to save
    bootstrap (int): Number of bootstrapped samples
    '''
    for axis_group in range(n_clusters):
        scale_bootstrapped_conts(axis_group, save_name, bootstrap=bootstrap, make_plot=make_plot)



def plot_spec_with_cont(axis_group, save_name, bootstrap_num=-1, sum_cont_df='None', use_sum_cont_df=False, make_plot=True):
    '''Plots the spectrum witht he continuum overlaid
    
    Parameters:
    axis_group (int): Group number
    save_name (str): Name of the location to save
    bootstrap_num (int): number of the current bootstrap
    sum_cont_df (pd.DataFrame): set to the sum cont df to avoid reading it in every time
    use_sum_cont_df (boolean): Set to True to use the provided sum cont df, False to read in a new one
    make_plot (boolean): Set to true to make and save a plot, false otherwise
    '''
    if use_sum_cont_df==False:
        sum_cont_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts/{axis_group}_summed_cont.csv').to_pandas()
    
    # Top if statement if using a bootstrapped spectrum, bottom otherwise
    if bootstrap_num>-1:
        spec_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_spectra_boots/{axis_group}_spectrum_{bootstrap_num}.csv').to_pandas()
    else:
        spec_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_spectra/{axis_group}_spectrum.csv').to_pandas()
    # spec_df, sum_cont_df = scale_continuum(spec_df, sum_cont_df)
    spec_df, sum_cont_df = scale_cont_to_lines(spec_df, sum_cont_df, line_regions, mask_regions)

    # Save the scaled continuum
    if bootstrap_num>-1:
        sum_cont_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts_boots/{axis_group}_summed_cont_{bootstrap_num}.csv', index=False)
    else:
        sum_cont_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts/{axis_group}_summed_cont.csv', index=False)

    if make_plot==False:
        return

    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(spec_df['wavelength'], spec_df['f_lambda'], color = 'orange')
    ax.plot(sum_cont_df['rest_wavelength'], sum_cont_df['f_lambda_scaled'], color = 'black')

    ax.set_xlabel('Wavelength ($\AA$)', fontsize=14)
    ax.set_ylabel('Flux', fontsize=14)
    ax.tick_params(labelsize=12)

    if bootstrap_num>-1:
        fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts_boots/{axis_group}_cont_spec_{bootstrap_num}.pdf')
    else:
        fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts/{axis_group}_cont_spec.pdf')

    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(spec_df['wavelength'], spec_df['f_lambda'], color = 'orange')
    ax.plot(sum_cont_df['rest_wavelength'], sum_cont_df['f_lambda_scaled'], color = 'black')

    ax.set_xlabel('Wavelength ($\AA$)', fontsize=14)
    ax.set_ylabel('Flux', fontsize=14)
    ax.set_xlim(6500, 6620)
    ax.set_ylim(np.percentile(spec_df['f_lambda'], 2), np.percentile(spec_df['f_lambda'], 99.5))
    ax.tick_params(labelsize=12)


    if bootstrap_num>-1:
        fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts_boots/{axis_group}_zoomha_{bootstrap_num}.pdf')
    else:
        fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts/{axis_group}_zoomha.pdf')
    plt.close('all')

def plot_all_spec_with_cont(n_clusters, save_name):
    for axis_group in range(n_clusters):
        print(f'Plotting group {axis_group}')
        plot_spec_with_cont(axis_group, save_name)




def scale_cont_to_lines(spec_df, cont_df, line_regions, mask_regions):
    '''Scales the continuum separately around each emission line region, then adds them back together and saves
    
    Parameters:
    spec_df (pd.Dataframe): spectrum
    cont_df (pd.Dataframe): continuum
    line_regions (list of tuples):each tuple corresponds to a region where you want the continuum scaled to
    mask_regions (list of tuples):each tuple corresponds is of the form (start, end) where to mask out lines
    '''
    cont_df['f_lambda_scaled'] = cont_df['f_lambda']
    spec_df['rest_wavelength'] = spec_df['wavelength']
    for line in line_regions:
        region_start = line[0]
        region_end = line[1]
        # Only use the region sections defined at the top of code when matching
        def get_region(df):
            middle_idx = np.logical_and(df['rest_wavelength']>region_start, df['rest_wavelength']<region_end)
            return middle_idx

        # Only use the region sections defined at the top of code when matching
        def apply_mask(df, mask_regions):
            unmasked_points = []
            for k in range(len(mask_regions)):
                unmasked_point = np.logical_not(np.logical_and(df['rest_wavelength']>mask_regions[k][0], df['rest_wavelength']<mask_regions[k][1]))
                if len(unmasked_points) == 0:
                    unmasked_points = [unmasked_point]
                else:
                    unmasked_points = [np.logical_and(unmasked_points[0], unmasked_point)]
            return df[unmasked_points[0]]

        # Clip the extreme points - only use the median 16-84th percentile across both for correlation
        def clip_extremes(df, cutoff_pct=(16, 84)):
            '''Gets the indicles that cut the dataframe to just the specified percentiles

            Parameters:
            df (pd.DataFrmae): Dataframe to cut, operates on the f_lambda column
            cutoff_pct (tuple): (low, high) percents to cut at 
            '''
            idx = np.logical_and(df['f_lambda']>np.percentile(df['f_lambda'], cutoff_pct[0]), df['f_lambda']<np.percentile(df['f_lambda'], cutoff_pct[1]))
            return idx
        
        region_idxs = get_region(cont_df)

        spec_df_masked = apply_mask(spec_df[region_idxs], mask_regions)
        cont_df_masked = apply_mask(cont_df[region_idxs], mask_regions)

        med_spec_idx = clip_extremes(spec_df_masked)
        med_cont_idx = clip_extremes(cont_df_masked)
        med_both_idx = np.logical_and(med_spec_idx, med_cont_idx)

        # a12, b12 = get_cross_cor(cont_df[region_idxs][med_both_idx], spec_df[region_idxs][med_both_idx])
        # cont_df[region_idxs]['f_lambda_scaled'] = cont_df[region_idxs]['f_lambda']/a12
        # print(f'Scale factor: {a12}')

        # scale by the medians
        cont_df.loc[region_idxs, 'f_lambda_scaled'] = cont_df[region_idxs]['f_lambda'] * np.median(spec_df_masked[med_both_idx]['f_lambda']) / np.median(cont_df_masked[med_both_idx]['f_lambda'])
    return spec_df, cont_df


def scale_continuum(spec_df, cont_df):
    '''Cross-correlates continuum and spectrum to get a scale factor to match the cont to the spec

    First, only use the region from 5000 to 7000 angstroms. Then, only use the 16th to 84th percentiles in that region to avoid lines
    
    Parameters:
    spec_df (pd.DataFrame): Dataframe containing the spectra, wavelengths must match cont_df
    cont_df (pd.DataFrame): Dataframe containing the continuum, wavelengths must match
    
    '''

    # Only use the middle section (5k to 7k angstroms) when matching
    def get_middle(df):
        middle_idx = np.logical_and(df['rest_wavelength']>5000, df['rest_wavelength']<7000)
        return middle_idx

    # Clip the extreme points - only use the median 16-84th percentile across both for correlation
    def clip_extremes(df, cutoff_pct=(16, 84)):
        '''Gets the indicles that cut the dataframe to just the specified percentiles

        Parameters:
        df (pd.DataFrmae): Dataframe to cut, operates on the f_lambda column
        cutoff_pct (tuple): (low, high) percents to cut at 
        '''
        idx = np.logical_and(df['f_lambda']>np.percentile(df['f_lambda'], cutoff_pct[0]), df['f_lambda']<np.percentile(df['f_lambda'], cutoff_pct[1]))
        return idx
    
    middle_idxs = get_middle(cont_df)

    med_spec_idx = clip_extremes(spec_df[middle_idxs])
    med_cont_idx = clip_extremes(cont_df[middle_idxs])
    med_both_idx = np.logical_and(med_spec_idx, med_cont_idx)


    a12, b12 = get_cross_cor(cont_df[middle_idxs][med_both_idx], spec_df[middle_idxs][med_both_idx])
    cont_df['f_lambda_scaled'] = cont_df['f_lambda']/a12
    print(f'Scale factor: {a12}')

    # scale by the medians
    cont_df['f_lambda_scaled'] = cont_df['f_lambda'] * np.median(spec_df[middle_idxs][med_both_idx]['f_lambda']) / np.median(cont_df[middle_idxs][med_both_idx]['f_lambda'])
    return spec_df, cont_df


# plot_all_spec_with_cont(8, 'both_ssfrs_4bin_median_2axis')
# plot_all_spec_with_cont(18)
# plot_spec_with_cont(1, 'both_ssfrs_4bin_median_2axis')
# scale_all_bootstrapped_conts(8, 'both_ssfrs_4bin_median_2axis', 10, make_plot=False)

plot_all_spec_with_cont(8, 'both_sfms_4bin_median_2axis_boot100')