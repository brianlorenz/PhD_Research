from astropy.io import ascii
import matplotlib.pyplot as plt
import numpy as np
from read_catalog import sed_loc, spec_loc
from scale_spec import scale_spec

def plot_sed():
    sed = ascii.read(sed_loc).to_pandas()
    fig, ax = plt.subplots(figsize=(8,8))
    ax.set_xscale('log')
    spectrum = ascii.read(spec_loc).to_pandas()
    plot_spectrum(spectrum, ax)
    ax.errorbar(sed['rest_wavelength'], sed['rest_f_lambda'], yerr=sed['err_rest_f_lambda'], color='black', marker='o', ls='None')
    ax.set_ylim(-0.4e-19,np.percentile(sed['rest_f_lambda'], 100))
    fig.savefig('/Users/brianlorenz/jwst_sfgalaxy/plots/sed_spec.pdf')


def plot_spectrum(spectrum_scaled, ax):
    ax.plot(spectrum_scaled['rest_wavelength'], spectrum_scaled['rest_flux_total_scaled'], color='orange', ls='-', marker='.', alpha=0.5)




plot_sed()