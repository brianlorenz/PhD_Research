from uncover_read_data import read_segmap, read_spec_cat, read_supercat, read_bcg_surface_brightness
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.nddata import Cutout2D
from astropy.io import fits, ascii
from astropy.table import Table
from astropy.wcs import WCS
import numpy as np
import pandas as pd

def create_bcg_brightness_flag():
    bcg_df = read_bcg_surface_brightness()  #0.01 might be a reasonable flag to use? Definitely everything above 0.07 is bad
    supercat_df = read_supercat()
    # Stopping here, no need to fully make a flag for it

def find_bcg_surface_birghtness(id_DR3, segmap, bcg_model):
    bcg_values = bcg_model[segmap == id_DR3]
    bcg_summed_flux = np.sum(bcg_values)
    area = len(bcg_values)
    bcg_surface_brightness = bcg_summed_flux / area
    return bcg_surface_brightness, bcg_summed_flux, area


def find_all_bcg_brightnesses():
    supercat_df = read_supercat()
    bcg_model, bcg_wcs = read_bcg_model()
    segmap, segmap_wcs = read_segmap()

    id_dr3s = []
    bcg_surface_brightnesses = []
    bcg_fluxes = []
    segmap_areas = []

    for i in range(len(supercat_df)):
        id_dr3 = supercat_df['id'].iloc[i]
        id_dr3s.append(id_dr3)
        # obj_ra = supercat_df['ra'].iloc[i] * u.deg
        # obj_dec = supercat_df['dec'].iloc[i] * u.deg
        # obj_skycoord = SkyCoord(obj_ra, obj_dec)

        print(f'Computing bcg surface brightness for id_DR3 = {id_dr3}')
        bcg_surface_brightness, bcg_summed_flux, area = find_bcg_surface_birghtness(id_dr3, segmap, bcg_model)
        bcg_surface_brightnesses.append(bcg_surface_brightness)
        bcg_fluxes.append(bcg_summed_flux)
        segmap_areas.append(area)
    
    bcg_brightness_df = pd.DataFrame(zip(id_dr3s, bcg_surface_brightnesses, bcg_fluxes, segmap_areas), columns=['id_dr3', 'bcg_surface_brightness', 'bcg_summed_flux', 'segmap_area'])
    bcg_brightness_df.to_csv('/Users/brianlorenz/uncover/Data/generated_tables/bcg_surface_brightness.csv', index=False)

def read_bcg_model():
    image_str = '/Users/brianlorenz/uncover/Catalogs/uncover_v7.2_abell2744clu_f277w_bcgs_models.fits'
    with fits.open(image_str) as hdu:
        bcg_model = hdu[0].data
        bcg_wcs = WCS(hdu[0].header)

    return bcg_model, bcg_wcs


if __name__ == "__main__":
    # find_all_bcg_brightnesses() #Will take most of day to run

    # create_bcg_brightness_flag()

    bcg_df = read_bcg_surface_brightness() 
    # breakpoint()