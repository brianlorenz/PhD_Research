from astropy.io import fits
from astropy.wcs import WCS
import matplotlib.pyplot as plt
import numpy as np


fname = '/Users/brianlorenz/uncover/Catalogs/psf_matched/uncover_v7.2_abell2744clu_f200w_block40_bcgs_wht_f444w-matched.fits'
with fits.open(fname) as hdu:
    image = hdu[0].data
    wcs = WCS(hdu[0].header)
    photflam = hdu[0].header['PHOTFLAM']
    photplam = hdu[0].header['PHOTPLAM']

fig, ax = plt.subplots(figsize=(20,20)) 
ax.imshow(image, vmin=np.percentile(image, 10), vmax=np.percentile(image, 75))
plt.show()
