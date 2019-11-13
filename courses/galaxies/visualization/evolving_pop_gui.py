# GUI to visualize how a galaxy spectrum evolves over time
import fsps
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, SpanSelector


os.environ["SPS_HOME"] = "/Users/galaxies-air/SPS_Conroy/fsps/"

loc = '/Users/galaxies-air/Desktop/Galaxies/visualization/'


class StellarPop:
    def __init__(self, age=0):
        # Generate the initial SSP in here
        self.ssps = []  # Main variable to contain running total of ssps
        # Format (ssp, age) for all ssps
        initial_ssp = self.get_ssp()
        # Running total of ssps, will iterate over this for every observable
        self.ssps.append(initial_ssp)
        self.create_plot()
        self.update_plot(self.ssps)

    def get_ssp(self, metallicity=0.0, age=0):
        # Generates a stellar population model with the given parameters
        '''
        metallicity - float - log(Z) in solar units (so 0.0 is solar metallicity)
        age - float - how long after galaxy formation to birth this stellar population

        -See what other parameters might be interesting to chagne
        '''
        ssp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,
                                     sfh=0, logzsol=metallicity, dust_type=2, dust2=0, imf_type=0)
        return (ssp, age)

    def create_plot(self):
        # Sets up the figure for first time use. Called from __init__
        '''

        '''
        self.fig = plt.figure(figsize=(10, 8))
        # Axis where the spectrum is
        self.ax_spec = self.fig.add_axes([0.12, 0.1, 0.7, 0.3])
        #self.ax2 = self.fig.add_axes([0.12, 0.6, 0.7, 0.3])

    def update_plot(self, ssps):
        # Updates the figure after a change to the ssps
        '''
        ssps - self.ssps variable that gets passed around. Stores all ssps and their ages
        '''

        # TEST THIS LINE
        spectra = [ssps[i][0].get_spectrum(
            tage=ssps[i][1]) for i in range(len(ssps))]
        total_spectrum = np.sum(spectra)


stellar_pop = StellarPop()
sys.exit()

stellar_populations = [fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,
                                              sfh=0, logzsol=i, dust_type=2, dust2=0, imf_type=0) for i in (-0.5, 0.0, 0.5)]

spectra = [stellar_populations[i].get_spectrum(tage=1) for i in range(3)]

spectra_df = pd.DataFrame()
spectra_df['wavelength'] = spectra[0][0]
spectra_df['zlow_t1'] = spectra[0][1]
spectra_df['zmid_t1'] = spectra[1][1]
spectra_df['zhigh_t1'] = spectra[2][1]

spectra2 = [stellar_populations[i].get_spectrum(tage=2) for i in range(3)]
spectra_df['zlow_t2'] = spectra2[0][1]
spectra_df['zmid_t2'] = spectra2[1][1]
spectra_df['zhigh_t2'] = spectra2[2][1]

spectra5 = [stellar_populations[i].get_spectrum(tage=5) for i in range(3)]
spectra_df['zlow_t5'] = spectra5[0][1]
spectra_df['zmid_t5'] = spectra5[1][1]
spectra_df['zhigh_t5'] = spectra5[2][1]

spectra14 = [stellar_populations[i].get_spectrum(tage=14) for i in range(3)]
spectra_df['zlow_t14'] = spectra14[0][1]
spectra_df['zmid_t14'] = spectra14[1][1]
spectra_df['zhigh_t14'] = spectra14[2][1]

spectra_df.to_csv(loc+'ssp_spectra.df')
'''Outputs a dataframe with columns:
wavelength - ranges from 91 to 10^8 Angstrom
'z' + metallicity + '_t' + age - spectrum of ssp with that metallicity and age
metallicity can be 'low', 'mid', or 'high' (-0.5, 0, 0.5 Zsun)
age can be 1Gyr, 2, 5, or 14
'''

spec_df = pd.DataFrame()

for i in np.arange(0, 15):
    spec_df['zlow_t' +
            str(i)] = stellar_populations[0].get_spectrum(tage=(i+0.001))[1]
    spec_df['zmid_t' +
            str(i)] = stellar_populations[1].get_spectrum(tage=(i+0.001))[1]
    spec_df['zhigh_t' +
            str(i)] = stellar_populations[2].get_spectrum(tage=(i+0.001))[1]

spec_df.to_csv(loc+'ssp_spectra_all.df')

ssp_3burst_df = pd.DataFrame()
ssp_3burst_df['t0'] = spec_df['zlow_t0']
ssp_3burst_df['t1'] = spec_df['zlow_t1']+spec_df['zmid_t0']
for i in np.arange(2, 15):
    ssp_3burst_df['t'+str(i)] = spec_df['zlow_t'+str(i)] + \
        spec_df['zmid_t'+str(i-1)] + spec_df['zhigh_t'+str(i-2)]

ssp_3burst_df.to_csv(loc+'ssp_3burst.df')
