

from lifelines import KaplanMeierFitter
import pandas as pd
import numpy as np
from full_phot_read_data import read_merged_lineflux_cat, read_final_sample, read_ha_sample, read_possible_sample, read_paper_df, read_phot_df


sample_df = read_ha_sample()

median_bin = [9, 9.5]

subsample_df = sample_df[np.logical_and(sample_df['mstar_50']>median_bin[0], sample_df['mstar_50']<median_bin[1])]

# masses = sample_df['mstar_50'].to_list()
# limits = (sample_df['PaBeta_snr']>5).to_numpy()*1
pab_ratios = subsample_df['lineratio_pab_ha']
limits = (subsample_df['PaBeta_snr']>5).to_numpy()*1

u_lim_value = 0.04
values = [2, 3, 3, 3, 5, 7, 10]
detected = [0, 0, 0, 0, 1, 1, 1] 

# Sample data: 'T' is the time to event or censoring, 'E' is the event indicator (1 if event, 0 if censored)
data = {
    'T': pab_ratios,
    'E': limits  # 0 for censored, 1 for event
}
# data = {
#     'T': values,
#     'E': detected  # 0 for censored, 1 for event
# }

df = pd.DataFrame(data)

# Create a KaplanMeierFitter object
kmf = KaplanMeierFitter()

# Fit the model with time and event status
kmf.fit(durations=df['T'], event_observed=df['E'])

# Print the survival probabilities
print("Survival Probabilities:")
print(kmf.survival_function_)


# Median estimate
median = kmf.median_survival_time_
# Confidence interval for median
ci = kmf.confidence_interval_survival_function_
print(f"Median estimate: {median}")
print(f"Cofidence estimate: {ci}")


# Percentiles you want
percentiles = [0.16, 0.5, 0.84]  # CDF percentiles

# Convert to survival function levels
survival_probs = [1 - p for p in percentiles]

# Interpolate inverse survival function to get percentile estimates
sf = kmf.survival_function_
sf = sf.reset_index().rename(columns={'timeline': 'value', 'KM_estimate': 'survival'})

# Interpolate to find values where survival crosses 0.84, 0.5, 0.16
value_at_percentile = np.interp(survival_probs, sf['survival'][::-1], sf['value'][::-1])

p16, median, p84 = value_at_percentile

print(f"Median: {median:.5f} (+{p84 - median:.5f}/-{median - p16:.5f})")

# Plot the Kaplan-Meier curve
kmf.plot_survival_function()
import matplotlib.pyplot as plt
# plt.title('Kaplan-Meier Curve with Censored Data')
plt.ylabel('Upper Limit Fraction')
plt.xlabel('PaB/Ha')
plt.xlim(0, 0.2)
plt.show()