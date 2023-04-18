import fsps
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, SpanSelector
import time




sp = fsps.StellarPopulation(zcontinuous=1,
                                add_neb_emission=1)
sp.params['logzsol'] = -1.0
sp.params['gas_logz'] = -1.0
sp.params['gas_logu'] = -2.5
wave, spec = sp.get_spectrum(tage=0.1, peraa=True)
spec_df = pd.DataFrame(zip(wave, spec), columns=['wavelength', 'spectrum'])

plt.plot(wave, spec)
plt.show()