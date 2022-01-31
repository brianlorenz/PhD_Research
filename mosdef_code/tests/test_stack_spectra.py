'''Unit tests for functions in cross_correlate.py'''

import numpy as np
import pandas as pd
import pytest
from stack_spectra import stack_spectra, norm_axis_stack
from astropy.io import ascii
import initialize_mosdef_dirs as imd

def test_stack_spectra():
    """
    Tests the ability of stack spectra to make a stack by genearting aritficial spectra and getting the expected result
    """

    print("Running test of stack spectra for axis ratio groups")
    print("Testing normalization")

    # An object with twice as much flux, should have half the normalization
    assert 2 * norm_axis_stack(200, 2) == norm_axis_stack(100, 2)

    # An object of the same brightness but three times as far should have 1/(1+z)^2 the normalization??? What should it have? Depends on z? How does lumdist work? 
    pass



    print("Generating artifical spectra")
    ar_df = ascii.read(imd.mosdef_dir + '/axis_ratio_data/Merged_catalogs/stack_spectra_test.csv').to_pandas()
    spec_1_df = make_spec_df([1, 1, 1, 2, 3, 1], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])

    stack_spectra(0, 'cluster_norm', axis_ratio_df=df, axis_group=axis_group, save_name=cluster_name)

    
def make_spec_df(spec_vals, err_vals):
    """Makes a fake spectrum dataframe given values
    
    Parameters:
    spec_vals (list): Values fro the spectrum points
    err_vals (list): Values for the uncertainties in those points
    """
    wave_vals = range(len(spec_vals))
    return pd.DataFrame(zip(wave_vals, spec_vals, spec_vals, err_vals), columns = ['rest_wavelength', 'f_lambda_norm', 'cont_norm', 'err_f_lambda_norm'])


if __name__ == "__main__":
    spec_1_df = make_spec_df([1, 1, 1, 2, 3, 1], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    test_stack_spectra()

