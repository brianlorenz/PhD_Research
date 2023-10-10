# Codes for any cosmological calculations

import numpy as np
# from astropy.cosmology import WMAP9 as cosmo
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)


def luminosity_to_flux(luminosities, redshift):
    '''
    Given a redshift, converts a luminosity (erg/s) to a flux (erg/s/cm^2). Uses WMAP9 cosmology

    Parameters:
    luminosities (np.array): Luminosities to convert, ideally in erg/s
    redshift (float): Redshift of the object, e.g. 1.7

    Returns:
    fluxes (np.array): output fluxes in units of luminosities/cm^2
    '''

    # Find the luminostiy distance, then convert to cm
    lum_dist = cosmo.luminosity_distance(redshift)
    lum_dist = lum_dist.to(u.cm)

    fluxes = luminosities / (4 * np.pi * lum_dist.value ** 2)

    return fluxes

def flux_to_luminosity(fluxes, redshift, flux_errs = np.array(-1)):
    '''
    Given a redshift, converts a  flux (erg/s/cm^2) to a luminosity (erg/s). Uses WMAP9 cosmology

    Parameters:
    fluxes (np.array): output fluxes in units of luminosities/cm^2
    redshift (float): Redshift of the object, e.g. 1.7

    Returns:
    luminosities (np.array): Luminosities to convert, ideally in erg/s
    flux_errs (np.array): Set to the symmetric error value to compute and return errors
    '''

    # Find the luminostiy distance, then convert to cm
    lum_dist = cosmo.luminosity_distance(redshift)
    lum_dist = lum_dist.to(u.cm)

    luminosities = fluxes * (4 * np.pi * lum_dist.value ** 2) 

    if flux_errs[0] != -1:
        luminosity_errs = flux_errs * (4 * np.pi * lum_dist.value ** 2) 
        return luminosities, luminosity_errs
    else:
        return luminosities

def flux_to_luminosity_factor(redshift):
    '''
    Given a redshift, returns the factor to multiply flux (erg/s/cm^2) to get a luminosity (erg/s). Uses WMAP9 cosmology

    Parameters:
    redshift (float): Redshift of the object, e.g. 1.7

    Returns:
    factor (np.array): factor to multiply by flux to get luminosity
    '''

    # Find the luminostiy distance, then convert to cm
    lum_dist = cosmo.luminosity_distance(redshift)
    lum_dist = lum_dist.to(u.cm)

    factor = (4 * np.pi * lum_dist.value ** 2) 

    return factor

