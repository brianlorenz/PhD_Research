from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np
from uncover_read_data import read_raw_spec


def plot_image():
    fname = '/Users/brianlorenz/uncover/Catalogs/psf_matched/uncover_v7.1_abell2744clu_f105w_bcgs_sci_f444w-matched.fits.gz'
    with fits.open(fname) as hdu:
        image = hdu[0].data
        wcs = WCS(hdu[0].header)
        photflam = hdu[0].header['PHOTFLAM']
        photplam = hdu[0].header['PHOTPLAM']
        print(photflam)
        breakpoint()
    fig, ax = plt.subplots(figsize=(20,20)) 
    ax.imshow(image, vmin=np.percentile(image, 10), vmax=np.percentile(image, 75))
    plt.show()

def plot_spectrum(id_msa):
    spec_df = read_raw_spec(id_msa)
    
    fig, ax = plt.subplots(figsize=(6,6)) 
    ax.plot(spec_df['wave'], spec_df['flux'], marker='None', ls='-', color='black')
    plt.show()

plot_spectrum(6645)