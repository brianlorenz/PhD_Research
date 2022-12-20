import numpy as np
import weightedstats as ws

def bootstrap_median(df, sfr_weigh=False, sfr_df=False):
    """Bootstrap an error on a median from a column of a pandas dataframe
    
    Parameters:
    df (pd.DataFrame): One column of the dataframe
    sfr_weigh (boolean): Set to true to weight the median by sfr
    sfrs (pd.DataFrame): Pass in a dataframe of the sfrs if using weighing

    Returns
    err_median (float): Bootstrapped median uncertainty
    """
    df = df[df>-98]
    n_samples = 100
    samples = [np.random.choice(df, size=len(df)) for i in range(n_samples)]
    medians = [np.median(sample) for sample in samples]
    median = np.median(df)
    err_median = np.std(medians)
    err_median_low, err_median_high = np.percentile(medians, [16,84])
    err_median_low = median - err_median_low
    err_median_high = err_median_high - median
    if sfr_weigh == True:
        median = ws.weighted_median(df, weights=sfr_df)
        median_boots = []
        for i in range(n_samples):
            sample_nums = np.random.choice(np.arange(len(df)), size=len(df))
            median_sample = ws.weighted_median(df.iloc[sample_nums], weights=sfr_df.iloc[sample_nums])
            median_boots.append(median_sample)
        err_median = np.std(median_boots)
        err_median_low, err_median_high = np.percentile(median_boots, [16,84])
        err_median_low = median - err_median_low
        err_median_high = err_median_high - median
            
    return median, err_median, err_median_low, err_median_high

