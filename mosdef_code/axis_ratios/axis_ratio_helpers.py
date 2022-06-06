import numpy as np

def bootstrap_median(df):
    """Bootstrap an error on a median from a column of a pandas dataframe
    
    Parameters:
    df (pd.DataFrame): One column of the dataframe

    Returns
    err_median (float): Bootstrapped median uncertainty
    """
    df = df[df>-98]
    n_samples = 10000
    samples = [np.random.choice(df, size=len(df)) for i in range(n_samples)]
    medians = [np.median(sample) for sample in samples]
    median = np.median(df)
    err_median = np.std(medians)
    err_median_low, err_median_high = np.percentile(medians, [16,84])
    err_median_low = median - err_median_low
    err_median_high = err_median_high - median
    return median, err_median, err_median_low, err_median_high

