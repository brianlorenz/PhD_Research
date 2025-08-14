import numpy as np
from lifelines import KaplanMeierFitter

# Data
lower_bounds = np.array([-0.2, -0.21, -0.13, -0.12, -0.05])
# lower_bounds = np.array([1, 2, 3, 4, 5])

upper_bounds = np.array([-0.01, -0.01, -0.129, -0.119, -0.049])
# upper_bounds = np.array([3, 3, 3.1, 4.1, 5.1])
events = [0, 0, 1, 1, 1]

lower_bounds = np.array([1, 2, 3, 4, 5, 6, 7])
upper_bounds = np.array([3, 4, 5, 6, 7, 8, np.inf]) # Use np.inf for right-censored events

# Fit Kaplan-Meier
kmf = KaplanMeierFitter()
# kmf.fit(durations=lower_bounds, event_observed=events)
# median = kmf.median_survival_time_
# print("Kaplan-Meier median:", median)

kmf.fit_interval_censoring(lower_bound=lower_bounds, upper_bound=upper_bounds)

# Get median survival time
# median = kmf.median_survival_time_
# print("Kaplan-Meier median:", median)

 # Plot the survival function
kmf.plot_survival_function()
import matplotlib.pyplot as plt # Assuming you want to display the plot
plt.title('Kaplan-Meier Estimate for Interval Censored Data')
plt.xlabel('Time')
plt.ylabel('Survival Probability')
plt.show()