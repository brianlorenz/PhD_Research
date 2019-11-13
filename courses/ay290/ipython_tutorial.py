import numpy as np
import pandas as pd

b_mag = 15
v_mag = '13.2'

magnitude_data = pd.DataFrame([[b_mag, v_mag]], columns=['B', 'V'])

bv_color = magnitude_data['B'][0] - float(magnitude_data['V'])

print(np.round(bv_color, 3))
