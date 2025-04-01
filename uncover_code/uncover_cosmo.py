from astropy import units as u
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3)

pixel_scale = 0.04 #arcsec per pix

def find_pix_per_kpc(redshift):
    kpc_per_arcsec = cosmo.angular_diameter_distance(redshift).to('kpc') / 206265  
    kpc_per_pixel = pixel_scale * kpc_per_arcsec.value
    pixels_for_1kpc = 1/kpc_per_pixel

    return pixels_for_1kpc

breakpoint()