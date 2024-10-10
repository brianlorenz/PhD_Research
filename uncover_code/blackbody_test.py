import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling.models import BlackBody
from astropy import units as u
from astropy.io import ascii
from astropy.visualization import quantity_support
import pandas as pd
from uncover_sed_filters import unconver_read_filters
from sedpy import observate
from uncover_make_sed import read_sed
from make_dust_maps import compute_cont_pct, make_3color
from uncover_read_data import read_supercat, read_raw_spec, read_spec_cat, read_segmap, read_SPS_cat
from scipy.optimize import curve_fit
from fit_emission_uncover import line_list
from filter_integrals import get_transmission_at_line


def generate_mock_lines(ha_pab_ratio, flux_multiplier = 1, blackbody_temp = 4000, add_ons3=''):
    add_ons = ''
    add_ons2 = ''
    if flux_multiplier != 1:
        add_ons = f'_flux_{flux_multiplier}'
    if blackbody_temp != 4000:
        add_ons2 = f'_temp_{blackbody_temp}'
    if blackbody_temp == 0:
        add_ons2 = f'_flat'
    ha_pab_ratio_name = f'{ha_pab_ratio}{add_ons}{add_ons2}{add_ons3}'

    c = 299792458 # m/s

    ha_wave = 6564.6
    ha_amp = 2.32e-19 * flux_multiplier
    ha_sigma = 50
    ha_flux  = ha_amp * ha_sigma * np.sqrt(2 * np.pi)
    print(f'ha_flux = {ha_flux}')
    ha_flux_jy = ha_flux / (1e-23*1e10*c / ((ha_wave)**2))
    print(f'ha_flux jy = {ha_flux_jy}')

    pab_wave = 12821.7
    pab_sigma = 25
    pab_flux  = ha_flux / ha_pab_ratio
    pab_amp = pab_flux / (pab_sigma * np.sqrt(2 * np.pi))
    print(f'pab_flux = {pab_flux}')
    pab_flux_jy = pab_flux / (1e-23*1e10*c / ((pab_wave)**2))
    print(f'pab_flux jy = {pab_flux_jy}')
 
    if blackbody_temp == 0:
        wav = np.arange(4000, 20000, 0.5) * u.AA
        wavelength = wav.value
        flux = np.ones(len(wavelength)) * 1.5e-6
    else:
        wavelength, flux = generate_blackbody(T = blackbody_temp)
    c = 299792458 # m/s
    flux_erg_aa = flux * (1e-23*1e10*c / (wavelength**2))
    
    flux_erg_aa = add_emission_line(wavelength, flux_erg_aa, ha_wave, ha_amp, ha_sigma)
    flux_erg_aa = add_emission_line(wavelength, flux_erg_aa, pab_wave, pab_amp, pab_sigma)
    if '_he' in add_ons3:
        he_wave = 12560.034
        he_sigma = 25
        he_flux  = pab_flux / 4
        he_amp = he_flux / (he_sigma * np.sqrt(2 * np.pi))
        print(f'he_flux = {he_flux}')
        he_flux_jy = he_flux / (1e-23*1e10*c / ((he_wave)**2))
        print(f'he_flux jy = {he_flux_jy}')
        flux_erg_aa = add_emission_line(wavelength, flux_erg_aa, he_wave, he_amp, he_sigma)

    flux_jy = flux_erg_aa / (1e-23*1e10*c / (wavelength**2))
    err_flux = np.zeros(len(flux_erg_aa))
    # plt.plot(wavelength, flux_erg_aa)
    # plt.show()
    mock_gal_df = pd.DataFrame(zip(wavelength, flux_jy, flux_erg_aa, err_flux), columns = ['rest_wave_aa', 'rest_flux_jy', 'rest_flux_erg_aa', 'err_rest_flux_erg_aa'])
    mock_gal_df.to_csv(f'/Users/brianlorenz/uncover/Data/mock_spectra/mock_ratio_{ha_pab_ratio_name}.csv', index=False)

def generate_blackbody(T = 3500):
    bb = BlackBody(temperature=T*u.K)
    wav = np.arange(4000, 20000, 0.5) * u.AA
    flux = bb(wav)
    wavelength = wav.value # AA
    flux = 4*np.pi*flux.value # Jy
    wavelength_ha_idx = np.logical_and(wavelength > 6550, wavelength < 6580)
    wavelength_pab_idx = np.logical_and(wavelength > 13000, wavelength < 13050)
    scale_flux = np.max(flux[wavelength_ha_idx]) / 5e-7
    flux = flux / scale_flux
    return wavelength, flux

def add_emission_line(wavelength, flux_aa, peak, amplitude, sigma):
    gaussian_yvals = gaussian_func(wavelength, peak, amplitude, sigma)
    flux_aa = flux_aa + gaussian_yvals
    
    return flux_aa

def gaussian_func(wavelength, peak_wavelength, amp, sig):
    """Standard Gaussian funciton

    Parameters:
    wavelength (pd.DataFrame): Wavelength array to fit
    peak_wavelength (float): Peak of the line in the rest frame [angstroms]
    amp (float): Amplitude of the Gaussian
    sig (float): Standard deviation of the gaussian [angstroms]

    Returns:
    """
    return amp * np.exp(-(wavelength - peak_wavelength)**2 / (2 * sig**2))

def integrate_spec(mock_name, id_msa, use_filt='None'):
    if id_msa == 0:
        redshift = 0
    else:
        ha_filters, ha_images, wht_ha_images, obj_segmap = make_3color(id_msa, line_index=0, plot=False)
        pab_filters, pab_images, wht_pab_images, obj_segmap = make_3color(id_msa, line_index=1, plot=False)
        ha_sedpy_name = ha_filters[1].replace('f', 'jwst_f')
        ha_sedpy_filt = observate.load_filters([ha_sedpy_name])[0]
        pab_sedpy_name = pab_filters[1].replace('f', 'jwst_f')
        pab_sedpy_filt = observate.load_filters([pab_sedpy_name])[0]

        ha_red_sedpy_name = ha_filters[0].replace('f', 'jwst_f')
        ha_red_sedpy_filt = observate.load_filters([ha_red_sedpy_name])[0]
        pab_red_sedpy_name = pab_filters[0].replace('f', 'jwst_f')
        pab_red_sedpy_filt = observate.load_filters([pab_red_sedpy_name])[0]
        ha_blue_sedpy_name = ha_filters[2].replace('f', 'jwst_f')
        ha_blue_sedpy_filt = observate.load_filters([ha_blue_sedpy_name])[0]
        pab_blue_sedpy_name = pab_filters[2].replace('f', 'jwst_f')
        pab_blue_sedpy_filt = observate.load_filters([pab_blue_sedpy_name])[0]
        ha_filters = ['f_'+filt for filt in ha_filters]
        pab_filters = ['f_'+filt for filt in pab_filters]
        sed_df = read_sed(id_msa)
        zqual_df = read_spec_cat()
        redshift = zqual_df[zqual_df['id_msa']==id_msa]['z_spec'].iloc[0]

    """Measure the line ratio just from integrating spectrum over the trasnmission curve"""
    spec_df = ascii.read(f'/Users/brianlorenz/uncover/Data/mock_spectra/{mock_name}.csv').to_pandas()
    wavelength = spec_df['rest_wave_aa'].to_numpy() * (1+redshift)
    f_lambda = spec_df['rest_flux_erg_aa'].to_numpy() / (1+redshift)
    f_jy = spec_df['rest_flux_jy'].to_numpy() * (1+redshift)
    filt_dict, filters = unconver_read_filters()
    filter_names = [sedpy_filt.name for sedpy_filt in filters]
    if use_filt != 'None':
        names = [f'ha_red_{use_filt}', f'ha_green_{use_filt}', f'ha_blue_{use_filt}', f'pab_red_{use_filt}', f'pab_green_{use_filt}', f'pab_blue_{use_filt}']
        filters = [read_sedpy_filt(name) for name in names]
        effective_waves_aa = [filt.wave_effective for filt in filters]
        ha_sedpy_filt = filters[1]
        pab_sedpy_filt = filters[4]
        ha_red_sedpy_filt = filters[0]
        pab_red_sedpy_filt = filters[3]
        ha_blue_sedpy_filt = filters[2]
        pab_blue_sedpy_filt = filters[5]
    else:
        effective_waves_aa = sed_df['eff_wavelength']*10000
        
    integrated_sed_abmag = observate.getSED(wavelength, f_lambda, filterlist=filters)
    integrated_sed_jy = 10**(-0.4*(integrated_sed_abmag-8.9)) 

    

    # Alternative fit with polynomial
    line_waves = [6565*(1+redshift), 12821.7*(1+redshift)]
    line_masks = [np.logical_or(wavelength<(line_waves[i]-1000), wavelength>(line_waves[i]+1000)) for i in range(2)]
    wave_masks = [np.logical_and(wavelength>(line_waves[i]-3000), wavelength<(line_waves[i]+3000)) for i in range(2)]
    full_masks = [np.logical_and(line_masks[i], wave_masks[i]) for i in range(2)]
    ha_p5 = np.poly1d(np.polyfit(wavelength[full_masks[0]], f_jy[full_masks[0]], 5))
    pab_p5 = np.poly1d(np.polyfit(wavelength[full_masks[1]], f_jy[full_masks[1]], 5))
    ha_p5_erg = np.poly1d(np.polyfit(wavelength[full_masks[0]], f_lambda[full_masks[0]], 5))
    pab_p5_erg = np.poly1d(np.polyfit(wavelength[full_masks[1]], f_lambda[full_masks[1]], 5))

     # sed_df['err_spec_scaled_flux'] = poly5(sed_df['err_flux'], popt[0], popt[1], popt[2], popt[3], popt[4], popt[5])
    
    ha_idxs = []
    pab_idxs = []
    if id_msa != 0:
        for ha_filt in ha_filters:
            ha_filt = ha_filt[2:]
            for index, item in enumerate(filter_names):
                if ha_filt in item:
                    ha_idxs.append(index)
        for pab_filt in pab_filters:
            pab_filt = pab_filt[2:]
            for index, item in enumerate(filter_names):
                if pab_filt in item:
                    pab_idxs.append(index)
    
    if use_filt != 'None':
        ha_idxs = [0, 1, 2]
        pab_idxs = [3, 4, 5] # Filters ordered as ha_cont, ha_line, ha_cont, pab_cont, pab_line, pab_cont
    wave_key = 'wave_aa'
    idx_flags = np.zeros(len(integrated_sed_jy))
    idx_flags[ha_idxs[0]] = 1
    idx_flags[ha_idxs[1]] = 2
    idx_flags[ha_idxs[2]] = 1
    idx_flags[pab_idxs[0]] = 3
    idx_flags[pab_idxs[1]] = 4
    idx_flags[pab_idxs[2]] = 3

    integrated_spec_df = pd.DataFrame(zip(effective_waves_aa, integrated_sed_jy, idx_flags), columns=['wave_aa', 'integrated_spec_flux_jy', 'use_filter_flag'])
    integrated_spec_df.to_csv(f'/Users/brianlorenz/uncover/Data/integrated_specs/{mock_name}_shifted_{id_msa}.csv', index=False)
    def fint_pct_int_spec_df(integrated_spec_df, filt_idxs):
        red_row = integrated_spec_df.iloc[filt_idxs[0]]
        green_row = integrated_spec_df.iloc[filt_idxs[1]]
        blue_row = integrated_spec_df.iloc[filt_idxs[2]]
        cont_percentile = compute_cont_pct(blue_row.wave_aa, green_row.wave_aa, red_row.wave_aa, blue_row.integrated_spec_flux_jy, red_row.integrated_spec_flux_jy)
        return cont_percentile, red_row, green_row, blue_row
    ha_cont_pct, ha_red_row, ha_green_row, ha_blue_row = fint_pct_int_spec_df(integrated_spec_df, ha_idxs)
    pab_cont_pct, pab_red_row, pab_green_row, pab_blue_row = fint_pct_int_spec_df(integrated_spec_df, pab_idxs)

    ha_redflux = integrated_sed_jy[ha_idxs[0]]
    ha_greenflux = integrated_sed_jy[ha_idxs[1]]
    ha_blueflux = integrated_sed_jy[ha_idxs[2]]
    pab_redflux = integrated_sed_jy[pab_idxs[0]]
    pab_greenflux = integrated_sed_jy[pab_idxs[1]]
    pab_blueflux = integrated_sed_jy[pab_idxs[2]]
    # pab_redflux = pab_redflux * 0.99

    ha_cont = np.percentile([ha_redflux, ha_blueflux], ha_cont_pct*100)
    pab_cont = np.percentile([pab_redflux, pab_blueflux], pab_cont_pct*100)

    ha_line_scaled_transmission = get_transmission_at_line(ha_sedpy_filt, line_list[0][1] * (1+redshift))
    pab_line_scaled_transmission = get_transmission_at_line(pab_sedpy_filt, line_list[1][1] * (1+redshift))
    correction_ratio = pab_line_scaled_transmission/ha_line_scaled_transmission
    print(correction_ratio)

    ha_line = ha_greenflux - ha_cont
    pab_line = pab_greenflux - pab_cont
    ha_line = ha_line / (1+redshift)**2 # Added here
    pab_line = pab_line / (1+redshift)**2 # Added here
    line_ratio_from_spec = ha_line/pab_line
    line_ratio_from_spec = line_ratio_from_spec / correction_ratio
    new_ratio_test = (ha_line / ha_line_scaled_transmission) / (pab_line / pab_line_scaled_transmission) / ((line_list[0][1] / line_list[1][1])**2)
    ha_line_newmethod = (ha_line / ha_line_scaled_transmission) 
    pab_line_newmethod = (pab_line / pab_line_scaled_transmission) 

   
    
    # # Recompute with polyfit continuum
    # ha_cont_fit = ha_p5(ha_green_row['eff_wavelength']*10000)[0]
    # pab_cont_fit = pab_p5(pab_green_row['eff_wavelength']*10000)[0]
    

    # Recompute integrating polyfit
    ha_integrated_poly_abmag = observate.getSED(wavelength, ha_p5_erg(wavelength), filterlist=[filters[ha_idxs[1]]])
    pab_integrated_poly_abmag = observate.getSED(wavelength, pab_p5_erg(wavelength), filterlist=[filters[pab_idxs[1]]])
    ha_cont_fit = 10**(-0.4*(ha_integrated_poly_abmag-8.9))[0]
    pab_cont_fit = 10**(-0.4*(pab_integrated_poly_abmag-8.9))[0]
    ha_line_fit = ha_greenflux - ha_cont_fit
    pab_line_fit = pab_greenflux - pab_cont_fit
    line_ratio_from_spec_fit = ha_line_fit/pab_line_fit
    line_ratio_from_spec_fit = line_ratio_from_spec_fit / correction_ratio
    new_ratio_from_spec_fit = (ha_line_fit / ha_line_scaled_transmission) / (pab_line_fit / pab_line_scaled_transmission) / ((line_list[0][1] / line_list[1][1])**2)
    new_ha_line_fit = (ha_line_fit / ha_line_scaled_transmission) 
    new_pab_line_fit = (pab_line_fit / pab_line_scaled_transmission) 


    emission_df = ascii.read(f'/Users/brianlorenz/uncover/Data/emission_fitting/{mock_name}_emission_fits.csv').to_pandas()
    line_ratio_emission_fit = emission_df['ha_pab_ratio'].iloc[0]
    c = 299792458 # m/s
    emission_df['flux_jy'] = emission_df['flux'] / (1e-23*1e10*c / ((emission_df['line_center_rest'])**2))
    ha_flux_fit = emission_df['flux_jy'].iloc[0] 
    pab_flux_fit = emission_df['flux_jy'].iloc[1]
    

    ha_compared_to_em = ha_line_newmethod / ha_flux_fit
    pab_compared_to_em = pab_line_newmethod / pab_flux_fit
    print(ha_compared_to_em)
    print(pab_compared_to_em)



    fig, axarr = plt.subplots(2, 1, figsize=(6,8))
    ax_ha = axarr[0]
    ax_pab = axarr[1]
    
    colors = ['red', 'green', 'blue']
    def plt_filters(sedpy_filters, ax, greenflux):
        for i in range(len(sedpy_filters)):
            sedpy_filt = sedpy_filters[i]
            # sedpy_name = filters[i].replace('f_', 'jwst_')
            # sedpy_filt = observate.load_filters([sedpy_name])[0]
            scale = np.max(sedpy_filt.transmission/greenflux)
            ax.plot(sedpy_filt.wavelength/1e4, sedpy_filt.transmission/scale, ls='-', marker='None', color=colors[i], lw=1)
    plt_filters([ha_red_sedpy_filt, ha_sedpy_filt, ha_blue_sedpy_filt], ax_ha, ha_greenflux)
    plt_filters([pab_red_sedpy_filt, pab_sedpy_filt, pab_blue_sedpy_filt], ax_pab, pab_greenflux)

    # breakpoint()

    for row in [ha_red_row, ha_green_row, ha_blue_row, pab_red_row, pab_green_row, pab_blue_row]:
        row[wave_key] = row[wave_key] / 10000
    ax_ha.plot(ha_red_row[wave_key], ha_redflux, color='red', ls='None', marker='o')
    ax_ha.plot(ha_green_row[wave_key], ha_greenflux, color='green', ls='None', marker='o')
    ax_ha.plot(ha_blue_row[wave_key], ha_blueflux, color='blue', ls='None', marker='o')
    ax_ha.plot(ha_green_row[wave_key], ha_cont, color='purple', ls='None', marker='o')
    ax_ha.plot(ha_green_row[wave_key], ha_cont_fit, color='orange', ls='None', marker='o')
    ax_pab.plot(pab_red_row[wave_key], pab_redflux, color='red', ls='None', marker='o')
    ax_pab.plot(pab_green_row[wave_key], pab_greenflux, color='green', ls='None', marker='o')
    ax_pab.plot(pab_blue_row[wave_key], pab_blueflux, color='blue', ls='None', marker='o')
    ax_pab.plot(pab_green_row[wave_key], pab_cont, color='purple', ls='None', marker='o')
    ax_pab.plot(pab_green_row[wave_key], pab_cont_fit, color='orange', ls='None', marker='o')
    for ax in axarr:
        ax.plot(wavelength/10000, f_jy, ls='-', color='red', marker='None')
        ax.plot(wavelength[full_masks[0]]/10000, f_jy[full_masks[0]], ls='-', color='black', marker='None')
        ax.plot(wavelength[full_masks[1]]/10000, f_jy[full_masks[1]], ls='-', color='black', marker='None')
        ax.set_xlabel('Wavelength (microns)')
        ax.set_ylabel('Flux (Jy)')
    ax_ha.set_ylim(0.8*np.min(f_jy[full_masks[0]]), 1.2*np.max(f_jy[wave_masks[0]]))
    ax_pab.set_ylim(0.8*np.min(f_jy[full_masks[1]]), 1.2*np.max(f_jy[wave_masks[1]]))
    ax_ha.plot(wavelength[full_masks[0]]/10000, ha_p5(wavelength[full_masks[0]]), ls='-', color='orange', marker='None')
    ax_pab.plot(wavelength[full_masks[1]]/10000, pab_p5(wavelength[full_masks[1]]), ls='-', color='orange', marker='None')
    ax_ha.set_xlim([ha_blue_row[wave_key]-0.1, ha_red_row[wave_key]+0.1]) # Need iloc for some cases, not others
    ax_pab.set_xlim([pab_blue_row[wave_key]-0.1, pab_red_row[wave_key]+0.1])
    ax_ha.text(0.03, 0.9, f'int spec ratio: {round(new_ratio_test, 2)}', transform=ax_ha.transAxes, color='purple')
    ax_ha.text(0.03, 0.82, f'int spec fit ratio: {round(new_ratio_from_spec_fit, 2)}', transform=ax_ha.transAxes, color='orange')
    ax_ha.text(0.03, 0.74, f'emisison fit ratio: {round(line_ratio_emission_fit, 2)}', transform=ax_ha.transAxes, color='black')
    # ax_ha.text(0.03, 0.68, f'new ratio: {round(new_ratio_test, 2)}', transform=ax_ha.transAxes, color='black')

    ax_pab.text(0.03, 0.90, f'ha flux intspec: {ha_line_newmethod:0.2e}                  pab flux intspec: {pab_line_newmethod:0.2e}', transform=ax_pab.transAxes, color='purple')
    ax_pab.text(0.03, 0.82, f'ha flux int-fit: {new_ha_line_fit:0.2e}                  pab flux int-fit: {new_pab_line_fit:0.2e}', transform=ax_pab.transAxes, color='orange')
    ax_pab.text(0.03, 0.74, f'ha flux em-fit: {ha_flux_fit:0.2e}                  pab flux em-fit: {pab_flux_fit:0.2e}', transform=ax_pab.transAxes, color='black')
    fig.savefig(f'/Users/brianlorenz/uncover/Figures/diagnostic_lineratio/mock_spectra/{mock_name}_shifted_{id_msa}_{use_filt}.pdf')
    plt.close('all')
    return line_ratio_from_spec, ha_line, pab_line


def make_filt(filt_name, wave_min, wave_max, filt_type='boxcar', redshift=1.87):
    """waves in micron
    """
    waves = np.arange(wave_min, wave_max, 0.0001) * 10000

    if filt_type == 'boxcar':
        trasms = np.ones(len(waves)) * 0.5
    if filt_type == 'slanted':
        trasms = np.linspace(1, 0.2, len(waves))    
    if filt_type == 'avoid_line':
        trasms = np.ones(len(waves))
        for i in range(len(line_list)):
            line_wave = (line_list[i][1]) * (1+redshift)
            if (wave_min*10000) < line_wave and line_wave < (wave_max*10000):
                line_idx = np.argmin(np.abs(waves-line_wave))
                trasms[line_idx-500:line_idx+500] = 0.3
        
    trasms[0:10] = 0
    trasms[-10:] = 0
    
                        
    filt_df = pd.DataFrame(zip(waves, trasms))
    filt_df.to_csv(f'/Users/brianlorenz/uncover/Data/filter_test/{filt_name}.par', index=False, sep=' ', header=False)


def read_sedpy_filt(filt_name):
    filt_path = f'/Users/brianlorenz/uncover/Data/filter_test/'
    filt = observate.load_filters([filt_name], directory=filt_path)
    return filt[0]


def make_filts_ha_pab(filt_type='boxcar'):
    """Makes boxcars around both of the lines for most galaxies
    """
    make_filt(f'ha_blue_{filt_type}', 1.60, 1.70, filt_type=filt_type)
    make_filt(f'ha_green_{filt_type}', 1.80, 2.0, filt_type=filt_type)
    make_filt(f'ha_red_{filt_type}', 2.05, 2.15, filt_type=filt_type)

    make_filt(f'pab_blue_{filt_type}', 3.05, 3.25, filt_type=filt_type)
    make_filt(f'pab_green_{filt_type}', 3.57, 3.77, filt_type=filt_type)
    make_filt(f'pab_red_{filt_type}', 3.8, 3.9, filt_type=filt_type)


# def make_filts_ha_pab(filt_type='boxcar'): #Rest frame
#     """Makes boxcars around both of the lines for most galaxies
#     """
#     make_filt(f'ha_blue_{filt_type}', 0.40, 0.50, filt_type=filt_type)
#     make_filt(f'ha_green_{filt_type}', 0.55, 0.75, filt_type=filt_type)
#     make_filt(f'ha_red_{filt_type}', 0.80, 0.90, filt_type=filt_type)

#     make_filt(f'pab_blue_{filt_type}', 1.00, 1.10, filt_type=filt_type)
#     make_filt(f'pab_green_{filt_type}', 1.18, 1.38, filt_type=filt_type)
#     make_filt(f'pab_red_{filt_type}', 1.40, 1.50, filt_type=filt_type)


# make_filts_ha_pab(filt_type='boxcar')
# make_filts_ha_pab(filt_type='slanted')
# make_filts_ha_pab(filt_type='avoid_line')
# filt = read_sedpy_filt('ha_blue')
# breakpoint()

# generate_mock_lines(15, flux_multiplier=1, blackbody_temp=0)
# integrate_spec('mock_ratio_15_flat', 0, use_filt='boxcar')
# integrate_spec('mock_ratio_15_flat', 47875, use_filt='boxcar')
# integrate_spec('mock_ratio_15_flat', 47875, use_filt='avoid_line')
# integrate_spec('mock_ratio_15_flat', 47875)

# integrate_spec('mock_ratio_15_flat', 47875, use_filt='slanted')
# integrate_spec('mock_ratio_15_flat_wideha', 47875, use_filt='boxcar')
# integrate_spec('mock_ratio_12_flat_wideha', 47875, use_filt='boxcar')
# integrate_spec('mock_ratio_15_flat_narrowha', 47875, use_filt='boxcar')
integrate_spec('mock_ratio_15_flat_with_he', 47875, use_filt='boxcar')




# generate_mock_lines(12, flux_multiplier=1, blackbody_temp=0, add_ons3='_wideha')
# generate_mock_lines(15, flux_multiplier=1, blackbody_temp=0, add_ons3='_narrowha')
# generate_mock_lines(15, flux_multiplier=1, blackbody_temp=0, add_ons3='_with_he')
# generate_mock_lines(15, flux_multiplier=1, blackbody_temp=0)
# generate_mock_lines(10, flux_multiplier=1, blackbody_temp=4000)
# generate_mock_lines(15, flux_multiplier=1, blackbody_temp=3000)
# generate_mock_lines(15, flux_multiplier=100)
# integrate_spec('mock_ratio_15', 47875)
# integrate_spec('mock_ratio_15_flux_100', 47875)
# integrate_spec('mock_ratio_15', 25774)
# integrate_spec('mock_ratio_15_flux_100', 25774)
# integrate_spec('mock_ratio_10', 47875)
# integrate_spec('mock_ratio_15_temp_3000', 47875)
