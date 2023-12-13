import numpy as np

# Conroy 2009
def dust2_to_AV(dust2):
    A_V = 2.51*1/np.log(10)*dust2
    return A_V
