# Tests to better understand the bootstrapped errors
import sys
import os
import string
import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.io import fits
import matplotlib.pyplot as plt
import initialize_mosdef_dirs as imd
import matplotlib.pyplot as plt

groupID = 7
save_name = 'norm_1_sn5_filtered'
n_boots = 100

files = [imd.axis_cluster_data_dir + f'/{save_name}/{save_name}_emission_fits_boots/{groupID}_emission_fits_{i}.csv' for i in range(n_boots)]
emission_dfs = [ascii.read(file).to_pandas() for file in files]
ha_fluxes = [emission_dfs[i].iloc[0]['flux'] for i in range(n_boots)]
hb_fluxes = [emission_dfs[i].iloc[1]['flux'] for i in range(n_boots)]
balmer_decs = [ha_fluxes[i] / hb_fluxes[i] for i in range(n_boots)]


# plt.hist(ha_fluxes)
# plt.show()

# plt.hist(hb_fluxes)
# plt.show()

# plt.hist(balmer_decs)
# plt.show()