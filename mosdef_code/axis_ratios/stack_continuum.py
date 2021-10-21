import glob
import os
import initialize_mosdef_dirs as imd
from astropy.io import ascii
from scipy import interpolate
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def stack_continuum(axis_group, save_name='halpha_norm'):
    '''Stacks all of the normalized continuum fast fits'''

    

    cont_folder = imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/{axis_group}_conts/'
    file_names = glob.glob(os.path.join(cont_folder, '*'))

    cont_dfs = [ascii.read(file).to_pandas() for file in file_names]
    interp_conts = [interpolate.interp1d(cont_df['rest_wavelength'], cont_df['f_lambda_norm'], fill_value=0, bounds_error=False) for cont_df in cont_dfs]

    spectrum_wavelength = np.arange(3000, 10000, 1)
    interp_fluxes = [interp_cont(spectrum_wavelength) for interp_cont in interp_conts]
    summed_flux = np.sum(interp_fluxes, axis=0)         
    sum_cont_df = pd.DataFrame(zip(spectrum_wavelength, summed_flux), columns = ['rest_wavelength', 'f_lambda'])
    sum_cont_df.to_csv(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts/{axis_group}_summed_cont.csv', index=False)

def stack_all_continuum(n_clusters, save_name='halpha_norm'):
    for axis_group in range(n_clusters):
        print(f'Stacking group {axis_group}')
        stack_continuum(axis_group, save_name='halpha_norm')



def plot_spec_with_cont(axis_group, save_name='halpha_norm'):
    '''Plots the spectrum witht he continuum overlaid'''

    sum_cont_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts/{axis_group}_summed_cont.csv')
    spec_df = ascii.read(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_spectra/{axis_group}_spectrum.csv')

    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(spec_df['wavelength'], spec_df['f_lambda'], color = 'orange')
    ax.plot(sum_cont_df['rest_wavelength'], sum_cont_df['f_lambda'], color = 'black')

    ax.set_xlabel('Wavelength ($\AA$)', fontsize=14)
    ax.set_ylabel('Flux', fontsize=14)
    ax.tick_params(labelsize=12)

    fig.savefig(imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_conts/summed_conts/{axis_group}_cont_spec.pdf')
    plt.close('all')

def plot_all_spec_with_cont(n_clusters, save_name='halpha_norm'):
    for axis_group in range(n_clusters):
        print(f'Plotting group {axis_group}')
        plot_spec_with_cont(axis_group, save_name=save_name)



# stack_all_continuum(18)
plot_all_spec_with_cont(18)