from mosdef_obj_data_funcs import read_composite_sed
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
import pandas as pd

# Measure beta using the fiters between these bands
beta_range = (1268, 2580)

def func_linear(log_wavelength, beta, c):
    return c + log_wavelength * beta


def measure_all_beta(n_groups):
    gridsize = 5
    fontsize = 14
    fig, axarr = plt.subplots(gridsize, gridsize, figsize=(40,40))

    j = 0
    betas = []
    groupIDs = []
    for groupID in range(n_groups):
        groupIDs.append(groupID)
        i = groupID%gridsize
        # Read in the composite spectrum
        sed = read_composite_sed(groupID)
        beta_idxs = np.logical_and(sed['rest_wavelength'] >= beta_range[0], sed['rest_wavelength'] <= beta_range[1])
        sed_beta = sed[beta_idxs]
        sed_beta = sed_beta[sed_beta['f_lambda'] > 0]
        sed_beta['log_wavelength'] = np.log10(sed_beta['rest_wavelength'])
        sed_beta['log_flux'] = np.log10(sed_beta['f_lambda'])

        
        popt, pcov = curve_fit(func_linear, sed_beta['log_wavelength'], sed_beta['log_flux'])
        print(popt)
        beta = popt[0]

        ax = axarr[j, i]
        ax.plot(sed_beta['log_wavelength'], sed_beta['log_flux'], color='black', marker='o', ls='None')
        ax.plot(sed_beta['log_wavelength'], func_linear(sed_beta['log_wavelength'], popt[0], popt[1]), color='red')
        ax.text(0.75, 0.95, f'Beta: {round(beta, 3)}', fontsize = fontsize, transform=ax.transAxes, color='red')
        ax.set_xlabel('log(Rest Wavelength)', fontsize = fontsize)
        ax.set_ylabel('log(Flux)', fontsize = fontsize)
        ax.set_title(f'Group {groupID}')

        betas.append(beta)
        if i == gridsize-1:
            j = j+1
    beta_df = pd.DataFrame(zip(groupIDs, betas), columns=['groupID', 'composite_beta'])
    beta_df.to_csv(imd.loc_composite_beta_df, index=False)
    fig.savefig(imd.cluster_dir + '/cluster_stats/beta_composite_measure.pdf')
    
measure_all_beta(23)