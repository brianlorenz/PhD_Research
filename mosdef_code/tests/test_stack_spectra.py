'''Unit tests for functions in cross_correlate.py'''

import numpy as np
import pandas as pd
import pytest
from stack_spectra import stack_spectra, norm_axis_stack, perform_stack
import matplotlib.pyplot as plt


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
    # ar_df = ascii.read(imd.mosdef_dir + '/axis_ratio_data/Merged_catalogs/stack_spectra_test.csv').to_pandas()
    spec_1_df = make_spec_df([1, 1, 1, 2, 3, 0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    spec_2_df = make_spec_df([1, 2, 0, 2, 4, 0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    spec_3_df = make_spec_df([1, 0, 0, 3, 3, 0], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    spec_4_df = make_spec_df([-1, 6, 3, 3, 6, 5], [0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    spec_dfs = [spec_1_df, spec_2_df, spec_3_df, spec_4_df]
    total_spec, total_cont, total_errs, number_specs_by_wave, norm_value_specs_by_wave = perform_stack('mean', spec_dfs, [1,2,3,1])
    # Our expected spectrum, remembering that 0s are not counted in the means
    expected_spec = np.array([0.5, 3, 2, 2.5, 4, 5])
    assert (total_spec == expected_spec).all()

    # fig, ax = plt.subplots(1, 1, figsize=(8,8))
    # for spec_df in spec_dfs:
    #     ax.plot(spec_df['rest_wavelength'], spec_df['f_lambda_norm'], marker='o', ls='-')
    # ax.plot(spec_1_df['rest_wavelength'], total_spec, color='black', marker='o', ls='-')
    # plt.show()


    total_spec, total_cont, total_errs, number_specs_by_wave, norm_value_specs_by_wave = perform_stack('median', spec_dfs, [1,1,1,1])
    # Our expected spectrum, remembering that 0s are not counted in the means
    expected_spec = np.array([1, 2, 2, 2.5, 3.5, 5])
    assert (total_spec == expected_spec).all()

    # Number of spectra that contribute to each point - should ignore zeros
    expected_numbers = np.array([4, 3, 2, 4, 4, 1])
    assert (number_specs_by_wave == expected_numbers).all()



    print('All tests passed for stack_spectra')

    

    
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

