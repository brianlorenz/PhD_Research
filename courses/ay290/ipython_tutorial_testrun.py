import numpy as np
import pandas as pd

b_mag = 15
v_mag = '13.2'

magnitude_data = pd.DataFrame([[b_mag, v_mag_float]], columns=['B', 'V'])

color = magnitude_data['B'][0] - float(magnitude_data['V'][0])

print(np.round(color, 3))
