'''Unit tests for functions in cross_correlate.py'''

import numpy as np
import pandas as pd
import pytest
from stack_spectra import stack_spectra
from astropy.io import ascii
import initialize_mosdef_dirs as imd

def test_stack_spectra():
    """
    Tests the ability of stack spectra to make a stack by genearting aritficial spectra and getting the expected result
    """

    print("Running test of stack spectra for axis ratio groups")
    print("Generating artifical spectra")
    ar_df = ascii.read(imd.mosdef_dir + '/axis_ratio_data/Merged_catalogs/stack_spectra_test.csv').to_pandas()


    stack_spectra(0, 'cluster_norm', axis_ratio_df=df, axis_group=axis_group, save_name=cluster_name)

    


if __name__ == "__main__":
    test_stack_spectra()