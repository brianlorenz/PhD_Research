from uncover_read_data import read_raw_spec, get_id_msa_list, read_supercat_newids, read_supercat
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
from matplotlib.colors import Normalize, LogNorm
from astropy.io import ascii


def find_all_extraction_regions(id_msa_list):
    centers = []
    amps = []
    sigs = []
    conts = []
    fwhms = []
    for id_msa in id_msa_list:
        supercat_df = read_supercat()
        supercat_new = read_supercat_newids()
        id_dr3 = supercat_df[supercat_df['id_msa'] == id_msa]['id'].iloc[0]
        id_redux = supercat_new[supercat_new['id_DR3']== id_dr3]['id_redux'].iloc[0]

        fit_center, fit_amp, fit_sig, fit_cont_height, fwhm_gaussian  = find_slit_extraction_region(id_msa)
        centers.append(fit_center) 
        amps.append(fit_amp)
        sigs.append(fit_sig)
        conts.append(fit_cont_height)
        fwhms.append(fwhm_gaussian)
    extraction_df = pd.DataFrame(zip(id_msa_list, centers, amps, sigs, conts, fwhms), columns=['id_msa', 'center', 'amp', 'sig', 'cont_level', 'fwhm']) 
    extraction_df.to_csv('/Users/brianlorenz/uncover/Data/generated_tables/extraction_df.csv', index=False)

def find_slit_extraction_region(id_msa, id_redux=-1):
    spec_2d = read_raw_spec(id_msa, read_2d=4, id_redux=id_redux) 
    spec_1d = read_raw_spec(id_msa, id_redux=id_redux)

    spec_sci = read_raw_spec(id_msa, read_2d=2, id_redux=id_redux) 
    spec_wht = read_raw_spec(id_msa, read_2d=3, id_redux=id_redux) 
    sci_mid = spec_sci[0:27,100:300]
    wht_mid = spec_wht[0:27,100:300]
    sci_mid = np.nan_to_num(sci_mid, nan=-1)
    wht_mid = np.nan_to_num(wht_mid, nan=-1)
    frame_concat = np.concatenate((sci_mid*10, wht_mid/np.median(wht_mid)), axis=1)
    fig, ax = plt.subplots()
    ax.imshow(frame_concat, cmap='Greys_r')
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/slit_extraction/sci_wht/{id_msa}_sci_wht.pdf')
    plt.close('all')

    # Find the column corresponding to 4.4um, since everything is matched to f440m
    idx_4_4_um = np.argmin(np.abs(spec_1d['wave']-4.4))
    column_to_fit = np.sum(spec_2d[:,idx_4_4_um-5:idx_4_4_um+5], axis=1)
    slit_pixels = np.arange(column_to_fit.shape[0])

    guess = [np.median(slit_pixels), np.max(column_to_fit), 4, np.percentile(column_to_fit, 10)] # peak_wave, amp, sig
    bounds_low = [np.min(slit_pixels), 0, 0.5, np.min(column_to_fit)]
    bounds_high = [np.max(slit_pixels), 100000, len(slit_pixels), np.max(column_to_fit)]
    bounds = (np.array(bounds_low), np.array(bounds_high))


    popt, pcov = curve_fit(gaussian_func_with_cont, slit_pixels, column_to_fit, guess, bounds=bounds)

    fit_center, fit_amp, fit_sig, fit_cont_height = popt
    gauss_fit = gaussian_func_with_cont(slit_pixels, fit_center, fit_amp, fit_sig, fit_cont_height)
    hires_pix = np.arange(0, column_to_fit.shape[0], 0.001)
    hires_gauss = gaussian_func_with_cont(hires_pix, fit_center, fit_amp, fit_sig, fit_cont_height)

    fwhm_gaussian = 2 * np.sqrt(2*np.log(2)) * fit_sig

    fig, ax = plt.subplots(figsize=(6,6))
    ax.step(slit_pixels, column_to_fit, color='black', label='Data')
    ax.step(slit_pixels, gauss_fit, color='orange', label='Fit')
    # ax.plot(hires_pix, hires_gauss, color='pink', label='Hires')
    # ax.hlines((fit_amp+2*fit_cont_height)/2, xmin=fit_center-fwhm_gaussian/2, xmax=fit_center+fwhm_gaussian/2, label='FWHM')

    ax.set_xlabel('Slit Pixels', fontsize=14)
    ax.set_ylabel('Flux', fontsize=14)
    ax.legend(fontsize=12)

    fig.savefig(f'/Users/brianlorenz/uncover/Figures/slit_extraction/gaussians/{id_msa}_gauss_extraction.pdf')

    return fit_center, fit_amp, fit_sig, fit_cont_height, fwhm_gaussian 

def gaussian_func_with_cont(wavelength, peak_wavelength, amp, sig, cont_height):
    """Standard Gaussian funciton

    Parameters:
    wavelength (pd.DataFrame): Wavelength array to fit
    peak_wavelength (float): Peak of the line in the rest frame [angstroms]
    amp (float): Amplitude of the Gaussian
    sig (float): Standard deviation of the gaussian [angstroms]

    Returns:
    """
    return amp * np.exp(-(wavelength - peak_wavelength)**2 / (2 * sig**2)) + cont_height


def add_extraction_offsets():
    offsets = [0, -0.3, 0, 2, -1, 0, 0, -2.5, -2, 0, 1, 0, -1, 1]
    extraction_df = ascii.read('/Users/brianlorenz/uncover/Data/generated_tables/extraction_df.csv').to_pandas()
    extraction_df['offset'] = offsets
    extraction_df.to_csv('/Users/brianlorenz/uncover/Data/generated_tables/extraction_df.csv', index=False)



if __name__ == "__main__":
    # find_slit_extraction_region(35436)

    id_msa_list = get_id_msa_list(full_sample=False, referee_sample=True)
    find_all_extraction_regions(id_msa_list)

    # add_extraction_offsets()