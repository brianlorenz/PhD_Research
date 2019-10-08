import fsps
import os
os.environ["SPS_HOME"] = "/Users/galaxies-air/SPS_Conroy/fsps/"

sp = fsps.StellarPopulation(compute_vega_mags=False, zcontinuous=1,
                            sfh=0, logzsol=0.0, dust_type=2, dust2=0.2)
sp.libraries

sdss_bands = fsps.find_filter('sdss')
print(sdss_bands)
a = sp.get_mags(tage=13.7, bands=sdss_bands)
