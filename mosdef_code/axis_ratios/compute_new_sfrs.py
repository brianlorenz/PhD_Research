# Re-compute sfrs using new method from Av, useful for galaxies with lower limits of SFR_CORR
from astropy.io import ascii
import pandas as pd
import numpy as np
import initialize_mosdef_dirs as imd
from cosmology_calcs import flux_to_luminosity
import matplotlib.pyplot as plt



def convert_ha_to_sfr():
    '''Performs the whole process of adding the new sfr column to mosdef_all_cats_v2'''
    # Read in the dataframe and isolate the relevant columns
    ar_df, halpha_fluxes, avs_stellar, redshifts = read_ha_and_av()

    # Convert ha to luminsoty
    halpha_lums = flux_to_luminosity(halpha_fluxes, redshifts)

    #Add in a factor to go from AV_Stellar to AV_HII
    avs_balmer = correct_av_for_dust(avs_stellar)

    # Shift the AV from 5500 angstroms to 6565
    avs_balmer_6565 = apply_dust_law(avs_balmer, target_wave=6565)

    # Get dust-corrected halpha
    intrinsic_halpha_lums = correct_ha_lum_for_dust(halpha_lums, avs_balmer_6565)

    # Derive SFR from Kennicutt 1998
    halpha_sfrs = ha_lum_to_sfr(intrinsic_halpha_lums, imf='Chabrier')

    # Append the new sfrs to the catalog
    # Save the catalog
    ar_df['halpha_sfrs'] = halpha_sfrs
    ar_df.to_csv(imd.loc_axis_ratio_cat, index=False )


def read_ha_and_av():
    """Reads in the catalog"""
    ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()
    halpha_fluxes = ar_df['ha_flux']
    avs_stellar = ar_df['AV']
    redshifts = ar_df['z']
    return ar_df, halpha_fluxes, avs_stellar, redshifts


def correct_av_for_dust(avs_stellar):
    """Find the formula in Reddy et al 2015"""
    avs_balmer = avs_stellar + 0
    return avs_balmer

def apply_dust_law(avs_balmer, target_wave=6565, R_V=4.05):
    """Apply a Calzetti dust law

    Parameters:
    avs_balmer (pd.DataFrame) = AV of the star forming regions, in the V band at 5500 angstroms
    target_wave (float) = wavelenght to correct the extinction to (angstrom)
    R_V (float): R_V value fromt he Calzetti law

    Returns:
    avs_balmer_target (pd.DataFrame): Extinction now shifted to target wavelength
    """
    target_wave_um = target_wave / 10**4
    # Compute k(lambda) using Calzetti
    if target_wave < 6360:
        k_lambda = 2.659*(-2.156 + 1.509/target_wave_um - 0.198/(target_wave_um**2) + 0.011/(target_wave_um**3))+R_V
    elif target_wave >= 6360:
        k_lambda = 2.659*(-1.857 + 1.040/(target_wave_um)) + R_V

    avs_balmer_target = (k_lambda * avs_balmer)/R_V

    return avs_balmer_target

def correct_ha_lum_for_dust(obs_luminosity, attenuation):
    """Corrects observed luminsoity for attenuation
    
    Parameters:
    obs_luminosity (float): Observed luminosity, erg/s
    attenuation (float): Amount of aattenuation    
    
    Returns:
    int_luminosity (float): Intrinsic luminosity
    """
    int_luminosity = obs_luminosity / (10**(-0.4 * attenuation))
    return int_luminosity

def ha_lum_to_sfr(intrinsic_halpha_lums, imf='Chabrier'):
    """Converts halpha luminosity to SFR, values taken from Brown et al 2017, SFR Indicators
    
    Parameters:
    intrinsic_halpha_lums (pd.DataFrame): Dust corrected halpha luminosities
    imf (str): Imf name used   
    
    Returns:
    sfr (pd.DataFrame): Star formation rate (m_sun/yr)
    """
    if imf == 'Chabrier':
        sfr = 1.2 * 10**(-41) * intrinsic_halpha_lums
    return sfr





def plot_sfrs():
    """Plots the old vs new sfrs"""
    ar_df, halpha_fluxes, avs_stellar, redshifts = read_ha_and_av()

    ar_df = ar_df[ar_df['z']>2]

    hb_flag_filt = ar_df['hb_detflag_sfr'] == 1
    
    fig, ax = plt.subplots(figsize=(8,8))

    ax.plot(ar_df[~hb_flag_filt]['sfr'], ar_df[~hb_flag_filt]['halpha_sfrs'], color='black', marker='o', ls='None', label = 'H$_\\beta$ > 3 sigma')
    #  ax.plot(ar_df[hb_flag_filt]['sfr'], ar_df[hb_flag_filt]['halpha_sfrs'], color='orange', marker='o', ls='None', label = 'H$_\\beta$ < 3 sigma')
    ax.plot((0.001, 10**5), (0.001, 10**5), ls='--', color='blue')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('SFR2', fontsize=14)
    ax.set_xlim(0.1, 10000)
    ax.set_ylim(0.1, 10000)
    ax.set_ylabel('SFR from H$_\\alpha$', fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=12)
    fig.savefig(imd.axis_output_dir + '/sfr_comparison.pdf')



convert_ha_to_sfr()
plot_sfrs()



