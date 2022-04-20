import numpy as np

def bootstrap_median(df):
    """Bootstrap an error on a median from a column of a pandas dataframe
    
    Parameters:
    df (pd.DataFrame): One column of the dataframe

    Returns
    err_median (float): Bootstrapped median uncertainty
    """
    df = df[df>-999]
    n_samples = 10000
    samples = [np.random.choice(df, size=len(df)) for i in range(n_samples)]
    medians = [np.median(sample) for sample in samples]
    err_median = np.std(medians)
    return err_median

