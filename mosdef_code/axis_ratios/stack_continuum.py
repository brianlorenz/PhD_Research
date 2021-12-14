import glob
import os
import initialize_mosdef_dirs as imd
from astropy.io import ascii
from scipy import interpolate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cross_correlate import get_cross_cor



def stack_continuum(axis_group, save_name):
    '''Stacks all of the normalized continuum fast fits'''

    

    cont_folder = imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/{axis_group}_conts/'
    file_names = glob.glob(os.path.join(cont_folder, '*'))

    cont_dfs = [ascii.read(file).to_pandas() for file in file_names]
    interp_conts = [interpolate.interp1d(cont_df['rest_wavelength'], cont_df['f_lambda_norm'], fill_value=0, bounds_error=False) for cont_df in cont_dfs]

    spectrum_wavelength = np.arange(3000, 10000, 1)
    interp_fluxes = [interp_cont(spectrum_wavelength) for interp_cont in interp_conts]
    summed_flux = np.mean(interp_fluxes, axis=0)         
    sum_cont_df = pd.DataFrame(zip(spectrum_wavelength, summed_flux), columns = ['rest_wavelength', 'f_lambda'])
    sum_cont_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts/{axis_group}_summed_cont.csv', index=False)

def stack_all_continuum(n_clusters, save_name):
    for axis_group in range(n_clusters):
        print(f'Stacking group {axis_group}')
        stack_continuum(axis_group, save_name)



def plot_spec_with_cont(axis_group, save_name):
    '''Plots the spectrum witht he continuum overlaid'''

    sum_cont_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts/{axis_group}_summed_cont.csv').to_pandas()
    spec_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_spectra/{axis_group}_spectrum.csv').to_pandas()

    spec_df, sum_cont_df = scale_continuum(spec_df, sum_cont_df)

    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(spec_df['wavelength'], spec_df['f_lambda'], color = 'orange')
    ax.plot(sum_cont_df['rest_wavelength'], sum_cont_df['f_lambda_scaled'], color = 'black')

    ax.set_xlabel('Wavelength ($\AA$)', fontsize=14)
    ax.set_ylabel('Flux', fontsize=14)
    ax.tick_params(labelsize=12)

    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts/{axis_group}_cont_spec.pdf')

    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(spec_df['wavelength'], spec_df['f_lambda'], color = 'orange')
    ax.plot(sum_cont_df['rest_wavelength'], sum_cont_df['f_lambda_scaled'], color = 'black')

    sum_cont_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts/{axis_group}_summed_cont.csv', index=False)

    ax.set_xlabel('Wavelength ($\AA$)', fontsize=14)
    ax.set_ylabel('Flux', fontsize=14)
    ax.set_xlim(6500, 6620)
    ax.set_ylim(np.percentile(spec_df['f_lambda'], 2), np.percentile(spec_df['f_lambda'], 99.5))
    ax.tick_params(labelsize=12)



    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts/{axis_group}_zoomha.pdf')
    plt.close('all')

def plot_all_spec_with_cont(n_clusters, save_name):
    for axis_group in range(n_clusters):
        print(f'Plotting group {axis_group}')
        plot_spec_with_cont(axis_group, save_name)


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


# stack_all_continuum(18)
# plot_all_spec_with_cont(18)