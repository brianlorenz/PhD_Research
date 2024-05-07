from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.io import fits
from astropy import units as u
from astropy.nddata import Cutout2D
from uncover_read_data import read_supercat
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb


def make_3color(id_msa): 
    obj_skycoord = get_coords(id_msa)

    filt_red = 'f444w'
    filt_green = 'f356w'
    filt_blue = 'f277w'

    image_red = get_cutout(obj_skycoord, filt_red)
    image_green = get_cutout(obj_skycoord, filt_green)
    image_blue = get_cutout(obj_skycoord, filt_blue)

    image = make_lupton_rgb(image_red.data, image_green.data, image_blue.data, stretch=0.5)
    plt.imshow(image)
    # plt.imshow(cutout.data, origin='lower')
    plt.show()
    
def get_coords(id_msa):
    supercat_df = read_supercat()
    row = supercat_df[supercat_df['id_msa']==id_msa]
    obj_ra = row['ra'].iloc[0] * u.deg
    obj_dec = row['dec'].iloc[0] * u.deg
    obj_skycoord = SkyCoord(obj_ra, obj_dec)
    return obj_skycoord

def load_image(filt):
    image_folder = '/Users/brianlorenz/uncover/Catalogs/psf_matched/'
    image_str = f'uncover_v7.2_abell2744clu_{filt}_bcgs_sci_f444w-matched.fits'
    with fits.open(image_folder+image_str) as hdu:
        image = hdu[0].data
        wcs = WCS(hdu[0].header)
        photflam = hdu[0].header['PHOTFLAM']
        photplam = hdu[0].header['PHOTPLAM']
    return image, wcs

def get_cutout(obj_skycoord, filt):
    image, wcs = load_image(filt)
    size = (100, 100)
    cutout = Cutout2D(image, obj_skycoord, size, wcs=wcs)
    return cutout

# make_3color(6291)
# make_3color(22755)
# make_3color(42203)
# make_3color(42213)