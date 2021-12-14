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
    halpha_sfrs = ha_lum_to_sfr(intrinsic_halpha_lums, imf='Hao_Chabrier')

    # Append the new sfrs to the catalog
    # Save the catalog
    ar_df['halpha_sfrs'] = halpha_sfrs
    ar_df.to_csv(imd.loc_axis_ratio_cat, index=False)


def add_use_sfr():
    """Adds the use_sfr column to the catalog, which will use the sfr_corr for most galaxies but the halpha_sfr for those without hbeta"""
    ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()
    
    # Set the new column to a all -999s
    ar_df['use_sfr'] = np.ones(len(ar_df)) * -999
    
    # Find where the Hbeta readings are good, use sfr_corr for that
    halpha_good = ar_df['ha_detflag_sfr'] == 0
    hbeta_good = ar_df['hb_detflag_sfr'] == 0
    use_corr = np.logical_and(halpha_good, hbeta_good)
    ar_df.loc[use_corr, 'use_sfr'] = ar_df[use_corr]['sfr_corr']

    print(len(ar_df[use_corr]))

    # Find where Halpha is good but Hbeta is not, and use the halpha sfrs for that
    use_hasfr = np.logical_and(halpha_good, ~hbeta_good)
    ar_df.loc[use_hasfr, 'use_sfr'] = ar_df[use_hasfr]['halpha_sfrs']

    print(len(ar_df[use_hasfr]))

    # ar_df = ar_df.rename(columns={"sfr": "sfr2", "err_sfr": "err_sfr2"})
    # ar_df = ar_df.rename(columns={"use_sfr": "sfr"})

    # ar_df.to_csv(imd.loc_axis_ratio_cat, index=False)



def read_ha_and_av():
    """Reads in the catalog"""
    ar_df = ascii.read(imd.loc_axis_ratio_cat).to_pandas()
    halpha_fluxes = ar_df['ha_flux']
    avs_stellar = ar_df['AV']
    redshifts = ar_df['z']
    return ar_df, halpha_fluxes, avs_stellar, redshifts

def filter_ar_df(ar_df):
    '''Cut the catalog down to only those galaxies that we are using'''
    ar_df = ar_df[ar_df['agn_flag'] == 0]
    ar_df = ar_df[ar_df['z_qual_flag'] == 7]
    ar_df = ar_df[ar_df['ha_detflag_sfr'] == 0]
    ar_df = ar_df[ar_df['hb_detflag_sfr'] == 0]

    return ar_df

def correct_av_for_dust(avs_stellar):
    """Find the formula in Reddy et al 2015"""
    avs_balmer = avs_stellar
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
    if imf == 'Hao_Chabrier':  ## Hao et al 2013
        sfr = 10**(-41.056) * intrinsic_halpha_lums
    if imf == 'Salpeter':
        sfr = 7.9 * 10**(-42) * intrinsic_halpha_lums
    return sfr





def plot_sfrs():
    """Plots the old vs new sfrs"""
    ar_df, halpha_fluxes, avs_stellar, redshifts = read_ha_and_av()



    hb_flag_filt = ar_df['hb_detflag_sfr'] == 1
    
    fig, ax = plt.subplots(figsize=(8,8))

    high_sfrs_bad =  ar_df[hb_flag_filt]['sfr'] > 100

    ax.plot(ar_df[~hb_flag_filt]['sfr2'], ar_df[~hb_flag_filt]['halpha_sfrs'], color='black', marker='o', ls='None', label = 'H$_\\beta$ > 3 sigma')
    ax.plot(ar_df[hb_flag_filt]['sfr2'], ar_df[hb_flag_filt]['halpha_sfrs'], color='orange', marker='o', ls='None', label = 'H$_\\beta$ < 3 sigma')
    ax.plot(ar_df[hb_flag_filt][high_sfrs_bad]['sfr2'], ar_df[hb_flag_filt][high_sfrs_bad]['halpha_sfrs'], color='mediumseagreen', marker='o', ls='None', label = 'H$_\\beta$ < 3 sigma')
    ax.plot((0.001, 10**5), (0.001, 10**5), ls='--', color='blue')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('SFR2', fontsize=14)
    ax.set_xlim(0.1, 10000)
    ax.set_ylim(0.1, 10000)
    ax.set_ylabel('SFR from H$_\\alpha$', fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=12)
    fig.savefig(imd.axis_output_dir + '/sfr_diagnostics/sfr_comparison.pdf')



    # Figure similar to the one in Sedona's paper
    def sedona_plot(ar_df=ar_df):
        fig, ax = plt.subplots(figsize=(9,8))
        
        ar_df = filter_ar_df(ar_df)
        ar_df = ar_df[ar_df['z']>=1.7]
        ar_df = ar_df[ar_df['z']<=2.4]
        sfr_type = 'sfr_corr' #sfr_corr or sfr as strings

        high_sfrs_bad =  ar_df[hb_flag_filt][sfr_type] > 100

        filter_999s = np.logical_and(ar_df[~hb_flag_filt]['halpha_sfrs'] > 0, ar_df[~hb_flag_filt][sfr_type] > 0)

        sfr_ratio = ar_df[~hb_flag_filt][filter_999s]['halpha_sfrs'] / ar_df[~hb_flag_filt][filter_999s][sfr_type]

        color_map = plt.cm.get_cmap('RdYlBu') 
        reversed_color_map = color_map.reversed() 
        sc = ax.scatter(ar_df[~hb_flag_filt][filter_999s]['log_mass'], np.log10(sfr_ratio) , c=ar_df[~hb_flag_filt][filter_999s]['AV'], cmap = reversed_color_map, edgecolors='black', vmin=0, vmax=1.4)
        cbar = plt.colorbar(sc)
        ax.plot((8, 12), (0, 0), ls='--', color='black')

        ax.set_xlabel('Stellar Mass', fontsize=14)
        ax.set_xlim(8.7, 11.6)
        ax.set_ylim(-2.2, 1.3)
        
        ax.set_ylabel('log10(SFR H$_\\alpha$ / SFR Balmer)', fontsize=14)
        cbar.set_label('FAST AV', fontsize=14)
        ax.tick_params(labelsize=12)
        fig.savefig(imd.axis_output_dir + '/sfr_diagnostics/sfr_comparison_vsmass.pdf')
        return
    sedona_plot()

    

    fig, ax = plt.subplots(figsize=(8,8))

    bins = np.arange(0, 500, 25)

    

    ax.hist(ar_df[~hb_flag_filt]['sfr2'], bins=bins, color='black', label = 'H$_\\beta$ > 3 sigma')
    ax.hist(ar_df[hb_flag_filt]['sfr2'], bins=bins, color='orange', alpha=0.5, label = 'H$_\\beta$ < 3 sigma')
    ax.hist(ar_df[hb_flag_filt][high_sfrs_bad]['sfr2'], bins=bins, color='mediumseagreen', alpha=1, label = 'H$_\\beta$ < 3 sigma')
    
    
    ax.set_xlabel('SFR2', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=12)
    fig.savefig(imd.axis_output_dir + '/sfr_diagnostics/sfr2_hist.pdf')


    ### DIagnostic plots for why sfr is so highsfr


    fig, ax = plt.subplots(figsize=(8,8))

    remove_999s_good = np.logical_and(ar_df[~hb_flag_filt]['err_hb_flux']>-999, ar_df[~hb_flag_filt]['hb_flux']>-999)
    remove_999s_bad = np.logical_and(ar_df[hb_flag_filt]['err_hb_flux']>-999, ar_df[hb_flag_filt]['hb_flux']>-999)

    ax.plot(ar_df[~hb_flag_filt][remove_999s_good]['err_hb_flux'], ar_df[~hb_flag_filt][remove_999s_good]['hb_flux'], color='black', marker='o', ls='None', label = 'H$_\\beta$ > 3 sigma')
    ax.plot(ar_df[hb_flag_filt][remove_999s_bad]['err_hb_flux'], ar_df[hb_flag_filt][remove_999s_bad]['hb_flux'], color='orange', marker='o', ls='None', label = 'H$_\\beta$ < 3 sigma')
    ax.plot(ar_df[hb_flag_filt][high_sfrs_bad]['err_hb_flux'], ar_df[hb_flag_filt][high_sfrs_bad]['hb_flux'], color='mediumseagreen', marker='o', ls='None', label = 'H$_\\beta$ < 3 sigma')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot((0.001*10**-18, 10**5), (0.003*10**-18, 3*10**5), ls='--', color='blue', label='Flux = 3*error')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel('Hbeta flux error', fontsize=14)
    ax.set_ylabel('Hbeta flux', fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=12)
    fig.savefig(imd.axis_output_dir + '/sfr_diagnostics/sfr_hbflux_hbflxuerr.pdf')

    fig, ax = plt.subplots(figsize=(8,8))

    remove_999s_good = np.logical_and(ar_df[~hb_flag_filt]['err_ha_flux']>-999, ar_df[~hb_flag_filt]['ha_flux']>-999)
    remove_999s_bad = np.logical_and(ar_df[hb_flag_filt]['err_ha_flux']>-999, ar_df[hb_flag_filt]['ha_flux']>-999)

    ax.plot(ar_df[~hb_flag_filt][remove_999s_good]['err_ha_flux'], ar_df[~hb_flag_filt][remove_999s_good]['ha_flux'], color='black', marker='o', ls='None', label = 'H$_\\beta$ > 3 sigma')
    ax.plot(ar_df[hb_flag_filt][remove_999s_bad]['err_ha_flux'], ar_df[hb_flag_filt][remove_999s_bad]['ha_flux'], color='orange', marker='o', ls='None', label = 'H$_\\beta$ < 3 sigma')
    ax.plot(ar_df[hb_flag_filt][high_sfrs_bad]['err_ha_flux'], ar_df[hb_flag_filt][high_sfrs_bad]['ha_flux'], color='mediumseagreen', marker='o', ls='None', label = 'H$_\\beta$ < 3 sigma')

    ax.set_xlabel('Halpha flux error', fontsize=14)
    ax.set_ylabel('Halpha flux', fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=12)
    ax.set_xlim(-0.1*10**(-17), 4*10**(-17))
    fig.savefig(imd.axis_output_dir + '/sfr_diagnostics/sfr_haflux_haflxuerr.pdf')



    fig, ax = plt.subplots(figsize=(8,8))

    remove_999s_good = np.logical_and(ar_df[~hb_flag_filt]['ha_flux']>-999, ar_df[~hb_flag_filt]['hb_flux']>-999)
    remove_999s_bad = np.logical_and(ar_df[hb_flag_filt]['ha_flux']>-999, ar_df[hb_flag_filt]['hb_flux']>-999)

    ax.plot(ar_df[~hb_flag_filt][remove_999s_good]['ha_flux'], ar_df[~hb_flag_filt][remove_999s_good]['hb_flux'], color='black', marker='o', ls='None', label = 'H$_\\beta$ > 3 sigma')
    ax.plot(ar_df[hb_flag_filt][remove_999s_bad]['ha_flux'], ar_df[hb_flag_filt][remove_999s_bad]['hb_flux'], color='orange', marker='o', ls='None', label = 'H$_\\beta$ < 3 sigma')
    ax.plot(ar_df[hb_flag_filt][high_sfrs_bad]['ha_flux'], ar_df[hb_flag_filt][high_sfrs_bad]['hb_flux'], color='mediumseagreen', marker='o', ls='None', label = 'H$_\\beta$ < 3 sigma')

    # 2.86 theory line
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.plot((0, 2.86*10**-(15)), (0, 1*10**(-15)), ls='--', color='blue', label='Balmer dec = 2.86')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel('Halpha flux', fontsize=14)
    ax.set_ylabel('Hbeta flux', fontsize=14)
    ax.legend(fontsize=14)
    ax.tick_params(labelsize=12)
    fig.savefig(imd.axis_output_dir + '/sfr_diagnostics/sfr_haflux_hbflux.pdf')



# convert_ha_to_sfr()
# add_use_sfr()
plot_sfrs()



