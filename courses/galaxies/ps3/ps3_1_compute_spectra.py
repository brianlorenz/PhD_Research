# Generates ssp models asked for in problem one, then saves the output spectra into a folder in the form of a pandas dataframe

import fsps
import os
import pandas as pd
os.environ["SPS_HOME"] = "/Users/galaxies-air/SPS_Conroy/fsps/"

loc = '/Users/galaxies-air/Desktop/Galaxies/ps3/'

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
