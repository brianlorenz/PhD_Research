from cosmology_calcs import cosmo
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
import numpy as np

def convert_re_to_kpc(res, err_res, zs):
    """Converts a half-light radius in arcseconds to a physical radius in kpc

    Parameters:
    res (df): Half light radii in arcsec
    err_res (df): Uncertaintines on the res
    zs (df): Redshift
    
    Returns:
    re_kpcs (df): Res converted to kpc
    err_re_kpcs (df): Uncertainties on the measurements
    """

    # Convert arcsec to radians
    angle_rad = 4.84814 * (10**(-6)) * res

    #Convert redshift to distance
    lum_dist = cosmo.luminosity_distance(zs)
    lum_dist = lum_dist.to(u.kpc)

    # Conversion
    re_kpcs = np.tan(angle_rad) * lum_dist.value

    # Pct errors
    pct_errors = err_res/res
    err_re_kpcs = pct_errors*re_kpcs

    return re_kpcs, err_re_kpcs


print(cosmo.arcsec_per_kpc_proper(1.3))
print(cosmo.arcsec_per_kpc_proper(2))