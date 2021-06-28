'''Unit tests for functions in cross_correlate.py'''

import numpy as np
import pandas as pd
import pytest
from cross_correlate import get_cross_cor


def test_get_cross_cor():
    """
    Tests the ability of get_cross_cor to properly correlate two arrays. Identical arrays must give zero, opposite arrays must give one. Also tests to see if it masks bad data properly
    """

    print("Running test")
    opposite_array = np.array([1, -1, 1, -1, 1])
    ones_array = np.ones(5)
    half_array = np.ones(5)*0.5
    bad_array = np.ones(5)*0.8
    bad_array[3] = -99

    opposite_df = pd.DataFrame(opposite_array, columns=['f_lambda'])
    ones_df = pd.DataFrame(ones_array, columns=['f_lambda'])
    half_df = pd.DataFrame(half_array, columns=['f_lambda'])
    bad_df = pd.DataFrame(bad_array, columns=['f_lambda'])

    a12_same, b12_same = get_cross_cor(ones_df, ones_df)
    a12_half, b12_half = get_cross_cor(ones_df, half_df)
    a12_diff, b12_diff = get_cross_cor(ones_df, opposite_df)
    a12_mask, b12_mask = get_cross_cor(bad_df, ones_df)

    assert a12_same == 1
    assert b12_same == 0
    assert a12_half == 2
    assert b12_half == 0
    assert a12_diff == 0.2
    assert b12_diff == pytest.approx(1, abs=0.5)
    assert a12_mask == 0.8
    assert b12_mask == 0


if __name__ == "__main__":
    test_get_cross_cor()